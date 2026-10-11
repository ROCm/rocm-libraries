# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""A bounded, deterministic set of GEMMs around a solution's size boundaries."""

from dataclasses import dataclass, replace
from heapq import heappop, heappush
from math import lcm
from random import Random

from .metadata import LibraryMetadata, _Solution
from .models import _BENCH_ACTIVATIONS, GemmCase, GemmProfile, GemmShape, bench_type


@dataclass
class _Dimension:
    minimum: int = 1
    maximum: int = 8192
    multiple: int = 1

    def at_least(self, value: int, *, multiple: int = 1) -> int | None:
        step = lcm(self.multiple, multiple)
        value = max(value, self.minimum)
        value = (value + step - 1) // step * step
        return value if value <= self.maximum else None

    def remainder_at_least(self, value: int, *, divisor: int, multiple: int = 1) -> int | None:
        """First legal value at or above ``value`` not divisible by ``divisor``."""
        step = lcm(self.multiple, multiple)
        if step % divisor == 0:
            return None
        result = self.at_least(value, multiple=multiple)
        if result is not None and result % divisor == 0:
            result = self.at_least(result + step, multiple=multiple)
        return result


def _conjuncts(predicate):
    if predicate["type"] == "And":
        for child in predicate["value"]:
            yield from _conjuncts(child)
    else:
        # In particular, the children of Or/Not are not independent bounds.
        yield predicate


def _bounds(solution: _Solution, max_dimension: int) -> dict[str, _Dimension]:
    dimensions = {
        name: _Dimension(maximum=max_dimension) for name in ("m", "n", "batch_count", "k")
    }
    named = {
        "Free0SizeMultiple": ("m", "multiple"),
        "Free1SizeMultiple": ("n", "multiple"),
        "BatchSizeMultiple": ("batch_count", "multiple"),
        "BoundSizeMultiple": ("k", "multiple"),
        "BatchSizeEqual": ("batch_count", "equal"),
        "LeadingFree0SizesGreaterOrEqual": ("m", "minimum"),
        "LeadingFree1SizesGreaterOrEqual": ("n", "minimum"),
    }
    indexed = {
        "SizeMultiple": "multiple",
        "SizeEqual": "equal",
        "ProblemSizeEqual": "equal",
        "SizeGreaterThan": "greater",
        "SizeLessThan": "less",
        "SizeInRange": "range",
    }
    for predicate in _conjuncts(solution.predicate):
        kind, value, index = predicate["type"], predicate.get("value"), predicate.get("index", 0)
        if kind in named:
            name, operation = named[kind]
            if index not in ((0, -1) if kind == "BoundSizeMultiple" else (0,)):
                raise ValueError(f"Invalid ordinary GEMM dimension: {predicate}")
        elif kind in indexed:
            if index not in range(4):
                raise ValueError(f"Invalid ordinary GEMM dimension: {predicate}")
            name, operation = tuple(dimensions)[index], indexed[kind]
        elif kind == "GlobalSplitUCheckMinK":
            name, operation = "k", "minimum"
            value = value[0] * value[1] if value[1] > 1 else 1
        else:
            continue  # Coupled, disjunctive and hardware predicates remain native checks.
        bound = dimensions[name]
        if operation == "multiple":
            if type(value) is not int or value <= 0:
                raise ValueError(f"Invalid size multiple: {predicate}")
            bound.multiple = lcm(bound.multiple, value)
        if operation in {"minimum", "greater", "equal", "range"}:
            lower = value["min"] if operation == "range" else value
            bound.minimum = max(bound.minimum, lower + (operation == "greater"))
        if operation in {"less", "equal", "range"}:
            upper = value["max"] if operation == "range" else value
            bound.maximum = min(bound.maximum, upper - (operation in {"less", "range"}))
    return dimensions


def _small_shapes(bounds: dict[str, _Dimension], profile: GemmProfile, max_data_bytes: int):
    """Enumerate the bounded dimensional domain in increasing GEMM volume.

    Every feasible shape is reachable by increasing one axis from the minimum.
    Storage grows monotonically along those edges, so over-budget neighbors
    need not be expanded. The frontier grows with the requested count, rather
    than materializing a Cartesian product of every dimension.
    """
    names = tuple(bounds)
    first = tuple(bound.at_least(1) for bound in bounds.values())
    frontier = [(first[0] * first[1] * first[2] * first[3], first)]
    seen = {first}
    while frontier:
        _, values = heappop(frontier)
        yield GemmShape(**dict(zip(names, values)))
        for i, bound in enumerate(bounds.values()):
            next_value = values[i] + bound.multiple
            if next_value > bound.maximum:
                continue
            neighbor = values[:i] + (next_value,) + values[i + 1 :]
            if neighbor in seen:
                continue
            seen.add(neighbor)
            shape = GemmShape(**dict(zip(names, neighbor)))
            if profile.data_bytes(shape) <= max_data_bytes:
                heappush(frontier, (shape.m * shape.n * shape.k * shape.batch_count, neighbor))


def _runtime_cases(cases: list[GemmCase], solution: _Solution, seed: int) -> list[GemmCase]:
    """Assign one setting combination per shape, without expanding the budget.

    Half the epilogue draws select no bias/activation; the other half choose
    uniformly among the supported nonempty combinations. Optional scales are
    independently enabled with probability 1/2. Required MX scales stay fixed.
    """
    profile = cases[0].profile
    # String seeding is stable across processes. Paths, revision and numeric
    # indices do not influence a solution's draws; aliases get the same settings.
    rng = Random(f"{seed}:{solution.name}")
    activations = set(_BENCH_ACTIVATIONS)
    has_activation_list = False
    alphas, betas = [1, 2], [0, 1]
    for predicate in _conjuncts(solution.predicate):
        kind, value = predicate["type"], predicate.get("value")
        if kind == "ActivationEnumWhiteList":
            activations &= set(value)
            has_activation_list = True
        elif kind in {"BetaZero", "BetaOne"}:
            betas = [beta for beta in betas if beta == int(kind == "BetaOne")]
        elif kind in {"AlphaValue", "BetaValue"} and value != "Any":
            # Other scalar values use Any; beta=0 has its own BetaZero predicate.
            allowed = {"One": 1, "NegativeOne": -1}
            values = alphas if kind == "AlphaValue" else betas
            values[:] = [scalar for scalar in values if scalar == allowed.get(value)]
    if not has_activation_list or profile.activation_type == "None":
        activations.clear()
    activations.add("None")  # Disabling activation bypasses its enum whitelist.
    if not alphas or not betas:
        raise ValueError(
            f"Solution {solution.index}: scalar restrictions exclude alpha=1/2, beta=0/1"
        )
    epilogues = [
        (bias, activation)
        for bias in ((False, True) if profile.use_bias else (False,))
        for activation in sorted(activations)
        if bias or activation != "None"
    ]
    bias_types = solution.problem.get("biasDataTypeWhiteList") or [profile.bias_type]
    bias_sources = solution.problem.get("biasSrcWhiteList") or [profile.bias_source]
    # Reuse the formatter's supported type set instead of adding another table.
    for dtype in bias_types:
        bench_type(dtype)
    if any(source not in {0, 1, 3} for source in bias_sources):
        raise ValueError("unsupported bias source")

    result = []
    for position, case in enumerate(cases):
        bias, activation = (
            rng.choice(epilogues) if epilogues and rng.random() >= 0.5 else (False, "None")
        )
        selected = replace(
            profile,
            use_bias=bias,
            bias_type=rng.choice(bias_types) if bias else profile.bias_type,
            bias_source=rng.choice(bias_sources) if bias else profile.bias_source,
            scale_ab=profile.scale_ab if rng.random() < 0.5 else "",
            scale_cd=profile.scale_cd and rng.random() < 0.5,
            scale_alpha_vector=profile.scale_alpha_vector and rng.random() < 0.5,
        )
        result.append(
            replace(
                case,
                profile=selected,
                alpha=alphas[position % len(alphas)],
                beta=betas[(position // len(alphas)) % len(betas)],
                activation=activation,
                activation_args=(
                    rng.choice(((0.0, 6.0), (-1.0, 1.0)))
                    if activation == "Clamp"
                    else case.activation_args
                ),
            )
        )
    return result


def candidate_gemms(
    library: LibraryMetadata,
    solution_index: int,
    *,
    max_cases: int = 10,
    seed: int = 1,
    max_dimension: int = 8192,
    max_data_bytes: int = 128 * 1024 * 1024,
) -> list[GemmCase]:
    """Return distinct candidate problems for one explicit solution index.

    The fixed priority is: minimum, short/long K reductions, K tail and split
    remainder, full/partial/multiple output tiles, then multiple batches.
    ``max_cases`` selects a prefix. After the primary cases, larger budgets
    exercise more boundary combinations, then additional small GEMMs. A shorter
    result means the entire domain allowed by the extracted dimensional bounds,
    packed layout and dimension/data caps has been enumerated. It does not mean
    every kernel path or every natively supported problem has been covered.

    Fixed split-K uses ``DepthU * GlobalSplitU`` as its reduction scale. For
    automatic split-K and Stream-K, the scale is DepthU alone: their actual
    per-workgroup work depends on the shape and hardware. No loop coverage or
    native support is inferred from a candidate's label.

    Matrices are packed and strided-batched. ``max_data_bytes`` bounds their
    combined A/B/C/D storage; workspace, bias and scalar/MX scale buffers are
    separate. Unsupported profiles and an empty feasible candidate set raise
    ``ValueError``. An index not loaded in ``library`` raises ``KeyError``.
    After selecting shapes, a local RNG seeded by ``seed`` and the solution
    name assigns epilogues and optional settings. Half the epilogue draws
    disable bias/activation; other draws choose supported nonempty combinations.
    Alpha/beta cycle through (1,0), (2,0), (1,1), (2,1), subject to recorded
    scalar restrictions. Settings vary within the shape budget and preserve
    prefixes. This function performs no file I/O or GPU calls.
    """
    if type(solution_index) is not int or solution_index < 0:
        raise ValueError("solution_index must be a nonnegative integer")
    if any(type(n) is not int or n <= 0 for n in (max_cases, max_dimension, max_data_bytes)):
        raise ValueError("Candidate limits must be positive integers")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    solution = library._solutions[solution_index]
    try:
        profile = GemmProfile.from_problem_type(solution.problem)
    except ValueError as error:
        raise ValueError(f"Solution {solution_index}: {error}") from error
    bounds = _bounds(solution, max_dimension)
    for axis, packing in profile.packing_multiples:
        bounds[axis].multiple = lcm(bounds[axis].multiple, packing)
    first = {name: bound.at_least(1) for name, bound in bounds.items()}
    if any(value is None for value in first.values()):
        raise ValueError(f"Solution {solution_index}: no positive shape within the dimension limit")

    tile_m, tile_n = solution.mapping["macroTile"][:2]
    depth, split = solution.mapping["depthU"], solution.mapping["globalSplitU"]
    if min(tile_m, tile_n, depth) <= 0:
        raise ValueError("Macro tile and DepthU must be positive")
    stream_k = solution.mapping.get("tileProcessingStrategy") == "StreamK"
    fixed_split = split > 1 and not stream_k
    scale = depth * split if fixed_split else depth
    scale_name = "DepthU × configured GSU" if fixed_split else "DepthU"
    cases, seen = [], set()

    def add(case_id, reason, **axes):
        if len(cases) == max_cases or any(value is None for value in axes.values()):
            return False
        fitted = {
            name: bound.at_least(axes.get(name, first[name])) for name, bound in bounds.items()
        }
        if any(value is None for value in fitted.values()):
            return False
        shape = GemmShape(**fitted)
        if shape in seen or profile.data_bytes(shape) > max_data_bytes:
            return False
        seen.add(shape)
        cases.append(
            GemmCase(
                solution_index=solution.index,
                solution_name=solution.name,
                kernel_name=solution.kernel_name,
                metadata_file=solution.metadata_file,
                code_objects=solution.code_objects,
                target=library.target,
                shape=shape,
                profile=profile,
                id=case_id,
                reason=reason,
            )
        )
        return True

    add("minimum", "Smallest shape satisfying the extracted necessary dimensional bounds")
    if not cases:
        raise ValueError(f"Solution {solution_index}: the minimum candidate exceeds the data limit")

    k_bound = bounds["k"]
    # Start at the first legal full reduction block. A minimum above DepthU
    # must not collapse every short-K proposal onto the same shape.
    full_k = k_bound.at_least(scale, multiple=scale)
    if full_k is not None:
        step = lcm(scale, k_bound.multiple)
        for offset in range(3):
            k = full_k + offset * step
            add(f"k_{offset + 1}", f"K = {k // scale} × {scale_name}", k=k)
    for blocks in (8, 4):
        if add(
            "k_long",
            f"Longer reduction: {blocks} × the first full K block",
            k=blocks * full_k if full_k is not None else None,
        ):
            break

    tail_k = k_bound.remainder_at_least(2 * scale + 1, divisor=depth)
    add("k_tail", "K has a nonzero remainder modulo DepthU", k=tail_k)
    if fixed_split:
        uneven_k = k_bound.remainder_at_least(2 * scale + depth, divisor=scale, multiple=depth)
        add(
            "split_remainder",
            "Full DepthU blocks divide unevenly between split groups",
            k=uneven_k,
        )

    reference_k = k_bound.at_least(2 * scale) or first["k"]
    full_m = bounds["m"].at_least(tile_m, multiple=tile_m)
    full_n = bounds["n"].at_least(tile_n, multiple=tile_n)
    add("tile", "M and N contain only full macro tiles", m=full_m, n=full_n, k=reference_k)

    edge_m = bounds["m"].remainder_at_least(tile_m + 1, divisor=tile_m)
    edge_n = bounds["n"].remainder_at_least(tile_n + 1, divisor=tile_n)
    if edge_m is not None or edge_n is not None:
        add(
            "mn_edge",
            "A partial output tile, combined with a K tail when permitted",
            m=edge_m if edge_m is not None else (full_m or first["m"]),
            n=edge_n if edge_n is not None else (full_n or first["n"]),
            k=tail_k or reference_k,
        )
    add(
        "multi_tile",
        "Multiple output tiles in M and N",
        m=bounds["m"].at_least(2 * tile_m, multiple=tile_m),
        n=bounds["n"].at_least(3 * tile_n, multiple=tile_n),
        k=reference_k,
    )
    add(
        "batch",
        "More than one strided batch",
        batch_count=max(2, first["batch_count"]),
        k=reference_k,
    )

    # Broaden coverage at larger budgets without multiplying a dense M/N/K
    # grid. Pair reduction boundaries with full/partial tiles, and vary each
    # output axis separately to include narrow and wide GEMMs.
    multipliers = [1, 2, 3] + [2**bit for bit in range(2, max_dimension.bit_length())]
    for multiplier in multipliers:
        if len(cases) == max_cases:
            break
        for offset in (-1, 0, 1):
            k = (
                k_bound.at_least(multiplier * full_k + offset * k_bound.multiple)
                if full_k is not None
                else None
            )
            for label, m, n, batch, purpose in (
                ("full", full_m, full_n, first["batch_count"], "full output tiles"),
                ("m_edge", edge_m, full_n, first["batch_count"], "a partial M tile"),
                ("n_edge", full_m, edge_n, first["batch_count"], "a partial N tile"),
                (
                    "edges_batch",
                    edge_m,
                    edge_n,
                    max(2, first["batch_count"]),
                    "partial M/N tiles and multiple batches",
                ),
            ):
                add(
                    f"boundary_k_{multiplier}_{offset}_{label}",
                    f"K near {multiplier} × the first aligned reduction length; {purpose}",
                    m=m,
                    n=n,
                    k=k,
                    batch_count=batch,
                )
            for axis, first_full in (("m", full_m), ("n", full_n)):
                if first_full is None:
                    continue
                axes = {"m": full_m, "n": full_n, "k": reference_k}
                axes[axis] = bounds[axis].at_least(
                    multiplier * first_full + offset * bounds[axis].multiple
                )
                add(
                    f"boundary_{axis}_{multiplier}_{offset}",
                    f"{axis.upper()} near {multiplier} × its first full-tile size; "
                    "the other output axis is fixed",
                    **axes,
                )

    if len(cases) < max_cases:
        for shape in _small_shapes(bounds, profile, max_data_bytes):
            add(
                f"additional_{len(cases)}",
                "Additional small GEMM after the boundary cases",
                m=shape.m,
                n=shape.n,
                k=shape.k,
                batch_count=shape.batch_count,
            )
            if len(cases) == max_cases:
                break
    return _runtime_cases(cases, solution, seed)
