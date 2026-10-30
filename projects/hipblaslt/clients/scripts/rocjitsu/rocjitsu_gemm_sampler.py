# Copyright Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""Sample gfx950 GEMMs without depending on the installed solution inventory."""

SAMPLER_VERSION = "gfx950-aligned-v1"

# Each family carries compatible operands, outputs, scale format and orientations.
# See projects/hipblaslt/docs/reference/api-reference.rst and the gfx950 MX tests.
NUMERICAL_FAMILIES = {
    "fp32": ("f32_r", ("f32_r",), 0, ("NN", "NT", "TN", "TT")),
    "fp16": ("f16_r", ("f16_r", "f32_r"), 0, ("NN", "NT", "TN", "TT")),
    "bf16": ("bf16_r", ("bf16_r",), 0, ("NN", "NT", "TN", "TT")),
    "fp8": ("f8_r", ("bf16_r",), 0, ("NN", "NT", "TN", "TT")),
    "mxfp8": ("f8_r", ("f32_r", "f16_r", "bf16_r"), 1001, ("TN",)),
    "mxfp4": ("f4_r", ("f32_r", "f16_r", "bf16_r"), 1001, ("TN",)),
}
EPILOGUES = (
    (False, "none"),
    (True, "none"),
    (False, "relu"),
    (True, "relu"),
    (False, "gelu"),
    (True, "gelu"),
)


def sample_numerics(rng):
    family = rng.choice(tuple(NUMERICAL_FAMILIES))
    operand, outputs, scale, _ = NUMERICAL_FAMILIES[family]
    output = rng.choice(outputs)
    return family, {
        "a_type": operand,
        "b_type": operand,
        "c_type": output,
        "d_type": output,
        "compute_type": "c_f32_r",
        "scaleA": scale,
        "scaleB": scale,
    }


def sample_shape(rng):
    """Independent uniform draws from the positive multiples of 32 below 500."""
    return {key: 32 * rng.randrange(1, 16) for key in ("M", "N", "K")}


def sample_layout(rng, family):
    trans_a, trans_b = rng.choice(NUMERICAL_FAMILIES[family][3])
    return {"transA": trans_a, "transB": trans_b, "batch_count": 1}


def sample_scalars(rng):
    # Alpha stays nonzero so every case exercises a GEMM dispatch.
    return {
        "alpha": rng.choice((-2, -1, 0.5, 1, 2)),
        "beta": rng.choice((-2, -1, 0, 0.5, 1, 2)),
    }


def sample_epilogue(rng):
    bias, activation = rng.choice(EPILOGUES)
    return {
        "bias_vector": bias,
        "bias_type": "f32_r",
        "activation_type": activation,
    }


def sample_gemm(rng):
    family, numerics = sample_numerics(rng)
    return {
        "family": family,
        **numerics,
        **sample_shape(rng),
        **sample_layout(rng, family),
        **sample_scalars(rng),
        **sample_epilogue(rng),
    }
