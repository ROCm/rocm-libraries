# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Reconstruct and sample a race-sweep plan from the tested artifact itself."""

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import zlib

POLICY = "tile-depth-v1"
CASES_PER_KERNEL = 4


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_library(path):
    import msgpack

    payload = path.read_bytes()
    if path.name.endswith(".zlib"):
        payload = zlib.decompress(payload)
    data = msgpack.unpackb(payload, raw=False)
    if not isinstance(data, dict):
        raise ValueError(f"Expected a library mapping: {path}")
    return data


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
        return value == device["device_id"]
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
        in {
            f"Contraction_l_A{a}_B{b}_Cijk_Dijk"
            for a in ("ilk", "lik")
            for b in ("jlk", "ljk")
        }
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
    Failure to generate a bounded case is recorded after sampling, not replaced.
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
            minimum[dimension] = max(
                minimum[dimension], value + (kind == "SizeGreaterThan")
            )
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
            n = fixed.get(
                d, math.ceil(max(wanted, minimum[d]) / multiple[d]) * multiple[d]
            )
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


class KernelSample:
    """Bounded, order-independent seeded random priorities over kernel names.

    Aliases do not get extra sampling weight. Choose their representative by a
    separate priority. Selection does not depend on backend or generated cases.
    """

    def __init__(self, count, seed):
        self.count, self.seed, self.entries = count, seed, {}

    def consider(self, library, solution):
        name = solution["kernelName"]
        rank = hashlib.sha256((self.seed + "\0" + name).encode()).hexdigest()
        alias = hashlib.sha256(
            (self.seed + "\0" + library.name + "\0" + str(solution["index"])).encode()
        ).hexdigest()
        if rank in self.entries and alias >= self.entries[rank][0]:
            return
        if len(self.entries) == self.count and rank > max(self.entries):
            return
        self.entries[rank] = (alias, library, solution)
        if len(self.entries) > self.count:
            del self.entries[max(self.entries)]

    def selected(self):
        if len(self.entries) != self.count:
            raise ValueError(
                f"Requested {self.count} distinct kernels; found only {len(self.entries)}"
            )
        return [self.entries[k][1:] for k in sorted(self.entries)]


def make_plan(library_dir, target, device, count, seed):
    library_dir = (
        library_dir / target if (library_dir / target).is_dir() else library_dir
    )
    # Prefer compressed metadata just as the native loader does; do not inventory
    # both representations of a shard if an artifact happens to contain both.
    libraries = {
        p.name.removesuffix(".zlib"): p
        for p in sorted(library_dir.glob(f"*{target}.dat*"))
        if p.name.endswith((".dat", ".dat.zlib"))
    }
    if not libraries:
        raise ValueError(f"No {target} packaged metadata under {library_dir}")
    counts, sample, inputs = Counter(), KernelSample(count, seed), {}
    names, eligible_names = set(), set()
    indices = set()
    for path in sorted(libraries.values()):
        inputs[path.name] = sha256(path)
        for solution in read_library(path).get("solutions", []):
            if solution["index"] in indices:
                raise ValueError(
                    f"Duplicate solution index {solution['index']} in {path}"
                )
            indices.add(solution["index"])
            counts["solutions"] += 1
            name = hashlib.sha256(solution["kernelName"].encode()).digest()
            names.add(name)
            if not matches_hardware(solution["hardwarePredicate"], target, device):
                counts["other_or_unknown_hardware"] += 1
            elif not ordinary_problem(solution["problemType"]):
                counts["outside_adapter_scope"] += 1
            else:
                counts["eligible_solutions"] += 1
                eligible_names.add(name)
                sample.consider(path, solution)
    counts.update(
        unique_kernel_names=len(names), eligible_kernel_names=len(eligible_names)
    )
    jobs = []
    for job_id, (metadata, s) in enumerate(sample.selected()):
        job = {
            "id": job_id,
            "library": str(metadata),
            "problem_type": s["problemType"],
            "hardware_predicate": s["hardwarePredicate"],
            "problem_predicate": s["problemPredicate"],
            "size_mapping": s["sizeMapping"],
            "solutions": [
                {"index": s["index"], "name": s["name"], "kernel": s["kernelName"]}
            ],
        }
        try:
            job["cases"] = derive_cases(s)
        except (ValueError, KeyError, IndexError, TypeError) as error:
            job["cases"] = [
                {"id": label, "shape": None}
                for label in ("tile", "multi_tile", "mn_edge", "k_remainder")
            ]
            job["planning_error"] = str(error)
        main_object = metadata.with_name(
            metadata.name.removesuffix(".zlib").removesuffix(".dat") + ".co"
        )
        helpers = [
            library_dir / f"Kernels.so-000-{target}.hsaco",
            library_dir / f"hipblasltTransform_{target}.hsaco",
        ]
        job["code_objects"] = [str(main_object)] + [
            str(p) for p in helpers if p.is_file()
        ]
        for p in map(Path, job["code_objects"]):
            if not p.is_file():
                job["planning_error"] = f"Missing selected code object: {p.name}"
            elif p.name not in inputs:
                inputs[p.name] = sha256(p)
        jobs.append(job)
    fingerprint = hashlib.sha256(
        json.dumps(inputs, sort_keys=True).encode()
    ).hexdigest()
    return {
        "policy": POLICY,
        "seed": seed,
        "target": target,
        "kernels": count,
        "planned_cases": count * CASES_PER_KERNEL,
        "inventory": dict(counts),
        "input_sha256": inputs,
        "artifact_fingerprint": fingerprint,
        "library_dir": str(library_dir),
        "jobs": jobs,
    }
