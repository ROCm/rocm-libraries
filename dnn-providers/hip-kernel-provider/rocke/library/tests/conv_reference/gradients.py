# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Small independent gradient oracles for direct numerical checks without Torch.

These checks compute an oracle at test time. They do not use the deterministic
pinned-forward certificate for atomic gradient outputs.
"""

from __future__ import annotations

import numpy as np

from .contract import Case, normalized_distance, reference_scale
from reference_common.numeric import decode, encode


def operand_shapes(case: Case, direction: str) -> dict[str, tuple[int, ...]]:
    if direction == "dgrad":
        return {"dy": case.output_shape, "w": case.input_shapes["b"]}
    if direction == "wgrad":
        return {"dy": case.output_shape, "x": case.input_shapes["a"]}
    raise ValueError(f"unsupported gradient direction: {direction}")


def output_shape(case: Case, direction: str) -> tuple[int, ...]:
    operand_shapes(case, direction)
    return case.input_shapes["a" if direction == "dgrad" else "b"]


def make_inputs(case: Case, direction: str, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.Generator(np.random.PCG64(seed))
    return {
        name: encode(rng.uniform(-1, 1, shape).astype(np.float32), case.dtype)
        for name, shape in operand_shapes(case, direction).items()
    }


def independent_reference(case: Case, direction: str, inputs: dict) -> np.ndarray:
    """Differentiate grouped cross-correlation over exact quantized operands.

    Accumulate in float64. Indexing is defined here independently of production
    convolution geometry, decomposition and workspace helpers.
    """
    shapes = operand_shapes(case, direction)
    if set(inputs) != set(shapes):
        raise ValueError("incorrect gradient operands")
    arrays = {}
    for name, shape in shapes.items():
        value = decode(inputs[name], case.dtype).astype(np.float64)
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"invalid gradient operand: {name}")
        arrays[name] = value
    out = np.zeros(output_shape(case, direction), dtype=np.float64)
    dy = arrays["dy"]
    cpg, kpg = case.C // case.groups, case.K // case.groups
    for group in range(case.groups):
        cs = slice(group * cpg, (group + 1) * cpg)
        ks = slice(group * kpg, (group + 1) * kpg)
        for ho in range(dy.shape[1]):
            for wo in range(dy.shape[2]):
                for y in range(case.Y):
                    hi = ho * case.sH - case.pH + y * case.dH
                    if not 0 <= hi < case.Hi:
                        continue
                    for x in range(case.X):
                        wi = wo * case.sW - case.pW + x * case.dW
                        if not 0 <= wi < case.Wi:
                            continue
                        if direction == "dgrad":
                            out[:, hi, wi, cs] += (
                                dy[:, ho, wo, ks] @ arrays["w"][ks, y, x, :]
                            )
                        else:
                            out[ks, y, x, :] += (
                                dy[:, ho, wo, ks].T @ arrays["x"][:, hi, wi, cs]
                            )
    return out


def check_output(case: Case, actual: np.ndarray, reference: np.ndarray) -> float:
    """Use the existing gradient suites' 5% relative max-error limit.

    The denominator is at least one, matching the pinned forward metric. No
    exact-output identity is required for nondeterministic atomic accumulation.
    """
    decoded = decode(actual, case.dtype)
    if decoded.shape != reference.shape:
        raise ValueError("wrong gradient output shape")
    error = normalized_distance(decoded, reference, reference_scale(reference))
    if error >= 0.05:
        raise AssertionError(f"gradient normalized error {error} exceeds 0.05")
    return error
