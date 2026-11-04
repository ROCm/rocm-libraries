# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Plan a race sweep over original solution indices in a tested artifact."""

import concurrent.futures
import functools
import hashlib
import json
import math
import multiprocessing
import re
from collections import Counter
from pathlib import Path

from tensilelite.Utilities.BinaryLibraryIO import iterSolutionRecords
from tensilelite.Utilities.gemm_problem_bench import UnsupportedBenchProblem, bench_yaml
from tensilelite.Utilities.gemm_problem_for_solution import (
    LibraryMetadata,
    UnsupportedSolution,
    _activation_whitelist,
    _size_predicates,
)

POLICY = "tile-depth-v1"
REGULAR_POLICY = "regular-v1"

# Keep these directed relationships aligned with ChipIdRegistry in
# tensilelite/include/Tensile/AMDGPUPredicates.hpp.
CHIP_FALLBACKS = {
    chip: (0x75A0,) for chip in (0x75B0, 0x75A2, 0x75B2, 0x75A3, 0x75B3, 0x75A8, 0x75B8)
}


def validate_profile(config, target):
    """Check that an explicitly supplied profile describes one device consistently.

    The topology lists every CU slot; a product can enable fewer of them. For
    example, MI300X has 304 active CUs in 8 XCDs of 40 slots.
    """
    device = config["vm"]["gpu"]["device"]
    if device["gfx_target_version"] != {"gfx942": 90402, "gfx950": 90500}[target]:
        raise ValueError("Profile architecture does not match the requested target")
    simds, per_cu = device["simd_count"], device["simd_per_cu"]
    if min(simds, per_cu) <= 0 or simds % per_cu:
        raise ValueError("Profile SIMD count must be a positive multiple of SIMD per CU")
    cus = simds // per_cu
    per_partition = (
        device["num_shader_engines"]
        * device["num_shader_arrays_per_engine"]
        * device["num_cu_per_sh"]
    )
    if per_partition <= 0:
        raise ValueError("Profile shader resources must be positive")
    if target == "gfx950" and device["device_id"] not in (0x75A0, *CHIP_FALLBACKS):
        raise ValueError("Unregistered gfx950 PCI chip ID in profile")

    def topology_cus(node):
        match = re.search(r"\[(\d+):(\d+)\]$", node["name"])
        repetitions = int(match[2]) - int(match[1]) if match else 1
        if repetitions <= 0:
            raise ValueError("Empty profile topology range")
        children = sum(topology_cus(n) for n in node.get("children", []))
        return repetitions * (1 if node["type"] == "compute_unit" else children)

    slots = topology_cus(config["topology"]["root"])
    if slots % per_partition:
        raise ValueError("Profile topology CU slots disagree with shader resources")
    if cus > slots:
        raise ValueError("Profile has more active CUs than topology CU slots")


def regular_problem(solution, directory, target, *, reduction_blocks=8, alpha=1, beta=0):
    record = {
        **solution,
        "sizePredicates": _size_predicates(solution["problemPredicate"]),
        "activationWhiteList": _activation_whitelist(solution["problemPredicate"]),
    }
    metadata = LibraryMetadata(directory, target, (solution["index"],), {solution["index"]: record})
    return metadata.choose_single_problem(
        solution["index"], reduction_blocks=reduction_blocks, alpha=alpha, beta=beta
    )


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def matches_hardware(predicate, target, device):
    kind, value = predicate["type"], predicate.get("value")
    if kind == "AMDGPU":
        return matches_hardware(value, target, device)
    if kind in {"And", "Or"}:
        results = [matches_hardware(p, target, device) for p in value]
        return all(results) if kind == "And" else any(results)
    if kind == "True":
        return True
    if kind == "Processor":
        return value == target
    if kind == "PciChipId":
        chip = device.get("device_id")
        return (
            target == "gfx950"
            and chip is not None
            and (value == chip or value in CHIP_FALLBACKS.get(chip, ()))
        )
    if kind == "CUCount":
        return value == device["simd_count"] // device["simd_per_cu"]
    return False


def ordinary_problem(problem):
    # Both adapters implement this same initial feature scope. Rejections are
    # counted in the manifest; unsupported features are never silently coerced.
    if any(
        problem.get(k, False)
        for k in (
            "groupedGemm",
            "fusedGemmA2A",
            "sparse",
            "useGradient",
            "useE",
            "useGateResidual",
            "mxBlockA",
            "mxBlockB",
            "useInitialStridesAB",
            "useInitialStridesCD",
            "swizzleTensorA",
            "swizzleTensorB",
            "mirrorDimsA",
            "mirrorDimsB",
        )
    ):
        return False
    allowed = {"Half", "BFloat16", "Float", "Float8", "BFloat8", "Int8", "Int32"}
    return (
        all(problem[k] in allowed for k in ("aType", "bType", "cType", "dType"))
        and problem["cType"] == problem["dType"]
        and problem.get("computeType", "Float") in {"Float", "Int32"}
        and problem.get("activationType", "None") in {"None", "All", "Hipblaslt_all"}
        and problem.get("useScaleAB", "") in {"", "Scalar", "Vector"}
        and problem.get("stridedBatched", True)
        and problem.get("operationIdentifier", "")
        in {f"Contraction_l_A{a}_B{b}_Cijk_Dijk" for a in ("ilk", "lik") for b in ("jlk", "ljk")}
    )


def conjuncts(predicate):
    if predicate["type"] == "And":
        for child in predicate["value"]:
            yield from conjuncts(child)
    else:
        yield predicate


def derive_cases(solution):
    """Generate four distinct tile/depth-relative candidates, keeping native checks.

    This handles common dimensional constraints, not every native predicate.
    The clients remain authoritative for layout, workspace and device support.
    Failure to generate a bounded case is recorded for the selected index.
    """
    mapping = solution["sizeMapping"]
    mt_m, mt_n = mapping["macroTile"][:2]
    depth = mapping["depthU"]
    if min(mt_m, mt_n, depth) <= 0:
        raise ValueError("Non-positive macro tile or DepthU")
    minimum, maximum, multiple = [1] * 4, [8192] * 4, [1] * 4
    fixed = {}
    for p in conjuncts(solution["problemPredicate"]):
        kind, value, index = p["type"], p.get("value"), p.get("index", 0)
        dimension = {
            "Free0SizeMultiple": 0,
            "Free1SizeMultiple": 1,
            "BatchSizeMultiple": 2,
            "BoundSizeMultiple": 3,
            "LeadingFree0SizesGreaterOrEqual": 0,
            "LeadingFree1SizesGreaterOrEqual": 1,
            "BatchSizeEqual": 2,
        }.get(kind, index)
        if kind in {
            "SizeMultiple",
            "Free0SizeMultiple",
            "Free1SizeMultiple",
            "BatchSizeMultiple",
            "BoundSizeMultiple",
        }:
            multiple[dimension] = math.lcm(multiple[dimension], value)
        elif kind in {
            "LeadingFree0SizesGreaterOrEqual",
            "LeadingFree1SizesGreaterOrEqual",
            "SizeGreaterThan",
        }:
            minimum[dimension] = max(minimum[dimension], value + (kind == "SizeGreaterThan"))
        elif kind == "SizeLessThan":
            maximum[dimension] = min(maximum[dimension], value - 1)
        elif kind in {"SizeEqual", "ProblemSizeEqual", "BatchSizeEqual"}:
            if dimension in fixed and fixed[dimension] != value:
                raise ValueError("Contradictory size equalities")
            fixed[dimension] = value
        elif kind == "GlobalSplitUCheckMinK" and value[1] > 1:
            minimum[3] = max(minimum[3], value[0] * value[1])
        elif kind in {"Or", "Not"}:
            raise ValueError(f"Case policy cannot solve {kind} size predicates")

    def fit(shape):
        result = []
        for d, wanted in enumerate(shape):
            n = fixed.get(d, math.ceil(max(wanted, minimum[d]) / multiple[d]) * multiple[d])
            if not minimum[d] <= n <= maximum[d] or n % multiple[d]:
                raise ValueError("Size constraints exceed bounded case policy")
            result.append(n)
        m, n, batch, k = result
        # Conservative host/device/reference buffer allowance, independent of
        # datatype. Workspace remains separately bounded in the native clients.
        if batch * (m * k + n * k + 2 * m * n) * 16 > 128 * 1024 * 1024:
            raise ValueError("Case exceeds 128 MiB estimated data-buffer budget")
        return result

    proposals = [
        ("tile", [mt_m, mt_n, 1, depth]),
        ("multi_tile", [2 * mt_m, 3 * mt_n, 1, 2 * depth]),
        ("mn_edge", [mt_m + multiple[0], mt_n + multiple[1], 1, 4 * depth]),
        ("k_remainder", [2 * mt_m, 2 * mt_n, 1, 4 * depth + multiple[3]]),
    ]
    cases = []
    for label, proposal in proposals:
        shape = fit(proposal)
        # Equalities/minima can collapse candidates. Preserve distinct cases by
        # growing an unfixed dimension; never silently run duplicates four times.
        for d in (3, 0, 1):
            if shape not in [c["shape"] for c in cases]:
                break
            if d not in fixed:
                shape[d] += max(multiple[d], depth if d == 3 else (mt_m, mt_n)[d])
                shape = fit(shape)
        if shape in [c["shape"] for c in cases]:
            raise ValueError("Cannot generate four distinct bounded cases")
        cases.append(
            {
                "id": label,
                "shape": shape,
                "m_tail": bool(shape[0] % mt_m),
                "n_tail": bool(shape[1] % mt_n),
                "k_tail": bool(shape[3] % depth),
            }
        )
    return cases


def metadata_shards(library_dir, target):
    """The target's library directory and its metadata shards, in sorted order."""
    library_dir = library_dir / target if (library_dir / target).is_dir() else library_dir
    # Prefer compressed metadata just as the native loader does; do not inventory
    # both representations of a shard if an artifact happens to contain both.
    libraries = {
        p.name.removesuffix(".zlib"): p
        for p in sorted(library_dir.glob(f"*{target}.dat*"))
        if p.name.endswith((".dat", ".dat.zlib"))
    }
    if not libraries:
        raise ValueError(f"No {target} packaged metadata under {library_dir}")
    return library_dir, sorted(libraries.values())


def _map_shards(function, paths, workers):
    if workers > 1 and len(paths) > 1:
        context = multiprocessing.get_context("fork")
        with concurrent.futures.ProcessPoolExecutor(min(workers, len(paths)), context) as pool:
            return list(pool.map(function, paths))
    return list(map(function, paths))


def _hardware_leaves(predicate, kind):
    if predicate["type"] == kind:
        yield predicate["value"]
    elif predicate["type"] in {"AMDGPU", "And", "Or"}:
        children = predicate["value"]
        for child in children if isinstance(children, list) else [children]:
            yield from _hardware_leaves(child, kind)


def _shard_hardware_predicates(path):
    return {
        json.dumps(solution["hardwarePredicate"], sort_keys=True)
        for _, solution in iterSolutionRecords(path)
    }


def derive_devices(library_dir, target, template, workers=1):
    """Emulated devices that together match every hardware predicate in a catalogue.

    The first device is the template. Each other device is the template with a
    PCI chip ID and active CU count taken from the predicates, added for a
    predicate that no earlier device matches. A predicate that no such device
    matches is left for the inventory to count as other hardware.
    """
    _, paths = metadata_shards(library_dir, target)
    predicates = sorted(set().union(*_map_shards(_shard_hardware_predicates, paths, workers)))
    predicates = [json.loads(p) for p in predicates]
    per_cu = template["simd_per_cu"]
    leaves = lambda kind: {v for p in predicates for v in _hardware_leaves(p, kind)}
    chips = [template.get("device_id")]
    if target == "gfx950":
        chips += sorted(leaves("PciChipId") - set(chips))
    cus = [template["simd_count"] // per_cu]
    cus += sorted(leaves("CUCount") - set(cus), reverse=True)
    devices = [template]
    for predicate in predicates:
        if any(matches_hardware(predicate, target, device) for device in devices):
            continue
        for chip in chips:
            for cu in cus:
                device = {
                    **template,
                    "device_id": chip,
                    "simd_count": cu * per_cu,
                    "marketing_name": (
                        f"{target} emulated"
                        + ("" if chip is None else f" PCI {chip:#x}")
                        + f", {cu} CUs"
                    ),
                }
                if matches_hardware(predicate, target, device):
                    devices.append(device)
                    break
            else:
                continue
            break
    return devices


def _shard_last_index(path):
    return max((solution["index"] for _, solution in iterSolutionRecords(path)), default=-1)


def catalogue_end(library_dir, target, workers=1):
    """One past the largest solution index in the target's catalogue."""
    _, paths = metadata_shards(library_dir, target)
    return max(_map_shards(_shard_last_index, paths, workers)) + 1


class EmptySelection(ValueError):
    """No catalogue index meets the selection."""


class _UnsupportedInInventory(Exception):
    """A selected solution that the inventory found outside the adapter scope."""


def _profile_for(solution, target, profiles):
    return next(
        (
            name
            for name, resources in profiles.items()
            if matches_hardware(solution["hardwarePredicate"], target, resources)
        ),
        None,
    )


def _inventory_shard(
    path,
    *,
    target,
    library_dir,
    policy,
    case_parameters,
    profiles,
    range_begin,
    range_end,
    indices=None,
):
    """Count one metadata shard's solutions and keep those selected.

    Each selected entry is (index, solution, profile, planned, eligible). For
    regular-v1, planned is the problem and its bench input, or the reason it is
    unsupported. ``first_eligible`` maps each eligible (kernel name digest,
    profile) to its smallest index.
    """
    counts, coverage_gaps = Counter(), Counter()
    names, eligible_names, shard_indices, selected = set(), set(), [], []
    first_eligible = {}
    for _, solution in iterSolutionRecords(path):
        index = solution["index"]
        shard_indices.append(index)
        counts["solutions"] += 1
        name = hashlib.sha256(solution["kernelName"].encode()).digest()
        names.add(name)
        profile = _profile_for(solution, target, profiles)
        planned, eligible = None, False
        if profile is None:
            counts["other_or_unknown_hardware"] += 1
        elif policy == REGULAR_POLICY:
            # Only explicitly unimplemented features are coverage gaps.
            # Missing fields and malformed metadata must fail preparation.
            try:
                problem = regular_problem(solution, library_dir, target, **case_parameters)
                planned = (problem, bench_yaml([problem]))
            except (UnsupportedSolution, UnsupportedBenchProblem) as error:
                counts["outside_adapter_scope"] += 1
                coverage_gaps.update(
                    error.reasons if isinstance(error, UnsupportedSolution) else (error.reason,)
                )
                planned = str(error)
            else:
                eligible = True
        elif not ordinary_problem(solution["problemType"]):
            counts["outside_adapter_scope"] += 1
        else:
            eligible = True
        if eligible:
            counts["eligible_solutions"] += 1
            eligible_names.add(name)
            key = (name, profile)
            first_eligible[key] = min(index, first_eligible.get(key, index))
        in_range = range_begin <= index and (range_end is None or index < range_end)
        if in_range and (indices is None or index in indices):
            selected.append((index, solution, profile, planned, eligible))
    return {
        "sha256": sha256(path),
        "indices": shard_indices,
        "counts": counts,
        "coverage_gaps": coverage_gaps,
        "names": names,
        "eligible_names": eligible_names,
        "first_eligible": first_eligible,
        "selected": selected,
    }


def make_plan(
    library_dir,
    target,
    device,
    range_begin=0,
    range_end=None,
    policy=POLICY,
    profiles=None,
    reduction_blocks=8,
    alpha=1,
    beta=0,
    workers=1,
    indices=None,
    one_index_per_kernel=False,
):
    """Plan the selected indices and inventory the whole catalogue.

    ``indices``, if given, selects only those indices within the range; each
    must be in the catalogue. With ``one_index_per_kernel``, an eligible index
    is omitted when a smaller eligible index in the catalogue has the same
    kernel name and hardware profile. ``workers`` processes inventory the
    metadata shards in parallel.
    """
    if type(range_begin) is not int or not 0 <= range_begin < (1 << 31):
        raise ValueError("Solution index range begin must be a nonnegative native index")
    if range_end is not None and (
        type(range_end) is not int or not range_begin < range_end <= (1 << 31)
    ):
        raise ValueError("Solution index range end must be greater than begin and at most 2**31")
    if policy not in (POLICY, REGULAR_POLICY):
        raise ValueError(f"Unknown case policy: {policy}")
    if type(reduction_blocks) is not int or reduction_blocks <= 0:
        raise ValueError("reduction_blocks must be a positive integer")
    try:
        finite_scalars = all(
            type(value) in (int, float) and math.isfinite(value) for value in (alpha, beta)
        )
    except OverflowError:
        finite_scalars = False
    if not finite_scalars:
        raise ValueError("alpha and beta must be finite real numbers")
    if policy != REGULAR_POLICY and (reduction_blocks, alpha, beta) != (8, 1, 0):
        raise ValueError("Reduction blocks and scalar overrides require regular-v1")
    case_parameters = (
        {"reduction_blocks": reduction_blocks, "alpha": alpha, "beta": beta}
        if policy == REGULAR_POLICY
        else None
    )
    library_dir, paths = metadata_shards(library_dir, target)
    profiles = profiles or {"default": device}
    inventory = functools.partial(
        _inventory_shard,
        target=target,
        library_dir=library_dir,
        policy=policy,
        case_parameters=case_parameters,
        profiles=profiles,
        range_begin=range_begin,
        range_end=range_end,
        indices=None if indices is None else frozenset(indices),
    )
    shards = _map_shards(inventory, paths, workers)
    counts, inputs, coverage_gaps = Counter(), {}, Counter()
    names, eligible_names, seen, requested = set(), set(), set(), {}
    first_eligible = {}
    for path, shard in zip(paths, shards):
        inputs[path.name] = shard["sha256"]
        repeated = Counter(shard["indices"])
        duplicates = seen.intersection(repeated) | {i for i, n in repeated.items() if n > 1}
        if duplicates:
            raise ValueError(f"Duplicate solution index {min(duplicates)} in {path}")
        seen.update(shard["indices"])
        counts.update(shard["counts"])
        coverage_gaps.update(shard["coverage_gaps"])
        names |= shard["names"]
        eligible_names |= shard["eligible_names"]
        for key, index in shard["first_eligible"].items():
            first_eligible[key] = min(index, first_eligible.get(key, index))
        requested.update((entry[1]["index"], (path, *entry[1:])) for entry in shard["selected"])
    if indices is not None:
        missing = {
            i for i in indices if range_begin <= i and (range_end is None or i < range_end)
        } - requested.keys()
        if missing:
            raise ValueError(f"Solution indices not in the catalogue: {sorted(missing)[:10]}")
    selected = [requested[i] for i in sorted(requested)]
    duplicates = 0
    if one_index_per_kernel:
        representative = lambda s, profile: first_eligible[
            hashlib.sha256(s["kernelName"].encode()).digest(), profile
        ]
        kept = [e for e in selected if not e[4] or representative(e[1], e[2]) == e[1]["index"]]
        duplicates, selected = len(selected) - len(kept), kept
    if not selected:
        raise EmptySelection("No solution indices in the requested range")
    counts.update(unique_kernel_names=len(names), eligible_kernel_names=len(eligible_names))
    jobs = []
    for job_id, (metadata, s, profile, planned, _) in enumerate(selected):
        job = {
            "id": job_id,
            "profile": profile,
            "library": str(metadata),
            "problem_type": s["problemType"],
            "hardware_predicate": s["hardwarePredicate"],
            "problem_predicate": s["problemPredicate"],
            "size_mapping": s["sizeMapping"],
            "solutions": [{"index": s["index"], "name": s["name"], "kernel": s["kernelName"]}],
        }
        try:
            if job["profile"] is None:
                job["planning_outcome"] = "EXCLUDED_HARDWARE"
                raise ValueError("Requested solution does not match the emulated hardware")
            if policy == REGULAR_POLICY:
                if isinstance(planned, str):
                    raise _UnsupportedInInventory(planned)
                problem, job["bench_yaml"] = planned
                job["target_dispatches_per_case"] = 1
                job["cases"] = [{"id": "regular", "shape": [problem.m, problem.n, 1, problem.k]}]
                if problem.grouped_members is not None:
                    job["cases"][0]["grouped_shapes"] = [
                        [m, n, 1, k] for m, n, k in problem.grouped_members
                    ]
                members = (
                    problem.grouped_members
                    if problem.grouped_members is not None
                    else ((problem.m, problem.n, problem.k),)
                )
                if sum(m * k + n * k + 2 * m * n for m, n, k in members) * 16 > 128 * 1024 * 1024:
                    raise ValueError("Case exceeds 128 MiB estimated data-buffer budget")
            elif not ordinary_problem(s["problemType"]):
                job["planning_outcome"] = "UNSUPPORTED_FEATURE"
                raise ValueError("Requested solution is outside the client adapter scope")
            else:
                job["cases"] = derive_cases(s)
        except (
            ValueError,
            KeyError,
            IndexError,
            TypeError,
            _UnsupportedInInventory,
        ) as error:
            job.setdefault(
                "planning_outcome",
                (
                    "UNSUPPORTED_FEATURE"
                    if isinstance(error, _UnsupportedInInventory)
                    else "INVALID_INPUT"
                ),
            )
            job["cases"] = [
                {"id": label, "shape": None}
                for label in (
                    ("regular",)
                    if policy == REGULAR_POLICY
                    else ("tile", "multi_tile", "mn_edge", "k_remainder")
                )
            ]
            job["planning_error"] = str(error)
        main_object = metadata.with_name(
            metadata.name.removesuffix(".zlib").removesuffix(".dat") + ".co"
        )
        helpers = [
            library_dir / f"Kernels.so-000-{target}.hsaco",
            library_dir / f"hipblasltTransform_{target}.hsaco",
        ]
        job["code_objects"] = [str(main_object)] + [str(p) for p in helpers if p.is_file()]
        for p in map(Path, job["code_objects"]):
            if not p.is_file():
                job["planning_error"] = f"Missing selected code object: {p.name}"
                job["planning_outcome"] = "MISSING_CODE_OBJECT"
            elif p.name not in inputs:
                inputs[p.name] = sha256(p)
        jobs.append(job)
    fingerprint = hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()
    return {
        "policy": policy,
        "case_parameters": case_parameters,
        "coverage_gaps": dict(coverage_gaps),
        "target": target,
        "selection": "solution-index-range" if indices is None else "solution-index-list",
        "solution_index_range": {"begin": range_begin, "end": range_end},
        **(
            {}
            if indices is None
            else {
                "solution_index_list_sha256": hashlib.sha256(
                    "".join(f"{i}\n" for i in sorted(set(indices))).encode()
                ).hexdigest()
            }
        ),
        "solutions": len(selected),
        "kernels": len({s["kernelName"] for _, s, *_ in selected}),
        "one_index_per_kernel": one_index_per_kernel,
        "omitted_kernel_duplicates": duplicates,
        "planned_cases": sum(len(job["cases"]) for job in jobs),
        "inventory": dict(counts),
        "input_sha256": inputs,
        "artifact_fingerprint": fingerprint,
        "library_dir": str(library_dir),
        "jobs": jobs,
    }


def make_batches(jobs, size, *, group_by_shard):
    """Order shards within each profile and keep one profile per native process."""
    if size < 1:
        raise ValueError("Batch size must be positive")
    if group_by_shard:
        # Shards occupy disjoint index ranges in the packaged lazy library.
        # Their first selected indices give the same ordering as their range starts.
        first_index = {}
        for job in jobs:
            index = job["solutions"][0]["index"]
            shard = job["library"]
            first_index[shard] = min(first_index.get(shard, index), index)
        jobs = sorted(jobs, key=lambda j: (first_index[j["library"]], j["library"]))
    groups = {}
    for job in jobs:
        groups.setdefault(job.get("profile", "default"), []).append(job)
    return [
        group[start : start + size]
        for group in groups.values()
        for start in range(0, len(group), size)
    ]
