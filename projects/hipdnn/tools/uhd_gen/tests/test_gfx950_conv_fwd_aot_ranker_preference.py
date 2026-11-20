# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The shipped `hipkernel:Gfx950ConvFwdAot` ranker (``sort_kernel_catalog``/tflops)
prefers a kernel configuration with no obviously wasteful difference over one with an
unambiguously worse one, scored through the project's own flatbuffer walk
(:func:`uhd_gen.evaluate._flatbuffer_scorer`), not a hand-rolled or deterministic
stand-in.

Nothing else exercised the trained ranker's relative preference between two candidates
for the same problem; every existing test either checks self-consistency (prepared vs.
plain) or the *deterministic* fallback score (``aotScore``, which explicitly "nothing
yet measures which wins where" for most family pairs -- see ``Gfx950ConvFwdNative.cpp``).
This test scores the real shipped `model.bin` artifact instead.

The two rows differ only in `$kernel.vector_size_c`: 8 (a full vectorized store) vs. 1
(scalar), everything else held identical -- a difference the model's own training
objective (measured tflops) should reward in one direction regardless of which
problem or family it was trained on, so the assertion does not depend on ground truth
this repository does not carry.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("flatbuffers")
pytest.importorskip("lightgbm")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")

from uhd_gen import train_uhd  # noqa: E402
from uhd_gen.evaluate import _flatbuffer_scorer  # noqa: E402

_HEURISTICS_ROOT = (
    Path(__file__).resolve().parents[5]
    / "dnn-providers"
    / "hip-kernel-provider"
    / "src"
    / "engines"
    / "kernel_ingestor_engine"
    / "descriptors"
    / "rocKE"
    / "gfx950_conv_fwd_aot"
    / "heuristics"
)


def _shipped_model_dirs() -> list[Path]:
    """Every `sort_kernel_catalog/default/tflops` directory the engine ships."""
    return sorted(_HEURISTICS_ROOT.glob("*/sort_kernel_catalog/default/tflops"))


def _real_forward_conv_graph_features() -> dict[str, float]:
    """`$graph.*` feature values of one real, physically consistent NCHW forward conv
    (a ResNet-50-shaped 3x3, stride 1, pad 1 layer: N=32, C=64, H=W=56, K=64), computed
    the same way hipDNN's own graph would carry them -- not the degenerate all-ones
    placeholder a first version of this test used, which let an unrelated tree split
    on a never-trained input region decide the comparison instead of the kernel-feature
    difference under test.
    """
    n, c, h, w, k, r, s, stride, pad, dilation = 32, 64, 56, 56, 64, 3, 3, 1, 1, 1
    h_out = (h + 2 * pad - dilation * (r - 1) - 1) // stride + 1
    w_out = (w + 2 * pad - dilation * (s - 1) - 1) // stride + 1

    def nchw_strides(dims):
        nn, cc, hh, ww = dims
        return (cc * hh * ww, hh * ww, ww, 1)

    def kcrs_strides(dims):
        kk, cc, rr, ss = dims
        return (cc * rr * ss, rr * ss, ss, 1)

    x_dims = (n, c, h, w)
    w_dims = (k, c, r, s)
    y_dims = (n, k, h_out, w_out)
    x_strides = nchw_strides(x_dims)
    w_strides = kcrs_strides(w_dims)
    y_strides = nchw_strides(y_dims)

    flops = 2 * n * k * c * r * s * h_out * w_out
    x_numel = n * c * h * w
    w_numel = k * c * r * s
    y_numel = n * k * h_out * w_out
    dtype_bytes = 2  # fp16/bf16
    logical_bytes = (x_numel + w_numel + y_numel) * dtype_bytes

    features: dict[str, float] = {
        "graph.arithmetic_intensity": flops / logical_bytes,
        "graph.flops": float(flops),
        "graph.flops_by_type.ConvolutionFwdAttributes": float(flops),
        "graph.logical_bytes": float(logical_bytes),
        "graph.nodes[0].dilation[0]": float(dilation),
        "graph.nodes[0].dilation[1]": float(dilation),
        "graph.nodes[0].flops": float(flops),
        "graph.nodes[0].post_padding[0]": float(pad),
        "graph.nodes[0].post_padding[1]": float(pad),
        "graph.nodes[0].pre_padding[0]": float(pad),
        "graph.nodes[0].pre_padding[1]": float(pad),
        "graph.nodes[0].stride[0]": float(stride),
        "graph.nodes[0].stride[1]": float(stride),
    }
    for prefix, dims, strides, numel in (
        ("graph.nodes[0].x", x_dims, x_strides, x_numel),
        ("graph.nodes[0].w", w_dims, w_strides, w_numel),
        ("graph.nodes[0].y", y_dims, y_strides, y_numel),
    ):
        for i, d in enumerate(dims):
            features[f"{prefix}.dims[{i}]"] = float(d)
        for i in (0, 2, 3):
            features[f"{prefix}.strides[{i}]"] = float(strides[i])
        features[f"{prefix}.numel"] = float(numel)
    # tensors[0..2] duplicate x/w/y in the graph's flat tensor table.
    for i, (dims, strides) in enumerate(
        ((x_dims, x_strides), (w_dims, w_strides), (y_dims, y_strides))
    ):
        for j, d in enumerate(dims):
            features[f"graph.tensors[{i}].dims[{j}]"] = float(d)
        for j in (0, 2, 3):
            features[f"graph.tensors[{i}].strides[{j}]"] = float(strides[j])
    return features


@pytest.mark.parametrize(
    "model_dir", _shipped_model_dirs(), ids=lambda p: p.parents[2].name
)
def test_vectorized_store_outranks_scalar_store_for_the_same_problem(model_dir):
    (descriptor_path,) = model_dir.glob("*.uhd.json")
    descriptor = json.loads(descriptor_path.read_text())
    signature = descriptor["features_signature"]
    features = [reference.removeprefix("$") for reference in signature]
    categorical_encoding = {
        reference.removeprefix("$"): codes
        for reference, codes in descriptor["categorical_encoding"].items()
    }
    graph_features = _real_forward_conv_graph_features()

    # A coherent, real implicit-GEMM kernel identity (not an arbitrary mix of
    # unrelated categories): `mem` pipeline, `cshuffle` epilogue, a real MFMA atom,
    # and no direct_variant (implicit GEMM has none).
    coherent_kernel = {
        "kernel.direct_variant": "none",
        "kernel.pipeline": "mem",
        "kernel.epilogue": "cshuffle",
        "kernel.atom": "32x32x16",
    }

    def value_of(feature: str):
        """A real graph's value for a `$graph.*` feature; a plausible shared value for
        every `$kernel.*` feature but `kernel.vector_size_c`; the data_type enum (not a
        declared category) as a constant, shared code."""
        if feature in graph_features:
            return graph_features[feature]
        if feature.endswith("data_type"):
            return 1.0
        if feature in coherent_kernel:
            return coherent_kernel[feature]
        if feature in categorical_encoding:
            # The lexicographically-first declared code: any valid category does, since
            # both rows share it.
            return min(
                categorical_encoding[feature], key=categorical_encoding[feature].get
            )
        if feature == "kernel.vector_size_c":
            return None  # filled in per-row below
        if feature in (
            "kernel.tile_m",
            "kernel.tile_n",
            "kernel.tile_k",
            "kernel.tile_h",
        ):
            return 64.0
        if feature in (
            "kernel.warp_m",
            "kernel.warp_n",
            "kernel.waves_m",
            "kernel.waves_n",
        ):
            return 2.0
        if feature in ("kernel.warp_tile_m", "kernel.warp_tile_n"):
            return 32.0
        if feature == "kernel.warp_tile_k":
            return 16.0
        if feature in ("kernel.vector_size_a", "kernel.vector_size_b"):
            return 8.0
        return 1.0

    base = {feature: value_of(feature) for feature in features}
    good, bad = dict(base), dict(base)
    good["kernel.vector_size_c"] = 8.0
    bad["kernel.vector_size_c"] = 1.0

    def encode(feature: str, value) -> float:
        if feature in categorical_encoding:
            return float(categorical_encoding[feature][value])
        return float(value)

    matrix = np.array(
        [[encode(f, good[f]) for f in features], [encode(f, bad[f]) for f in features]]
    )
    scorer_model_bin = model_dir / "model.bin"
    assert scorer_model_bin.exists(), f"no shipped artifact at {scorer_model_bin}"

    import uhd_gen  # noqa: F401  puts _generated/ on sys.path

    monkeypatch_target = train_uhd
    original = monkeypatch_target.build_feature_matrix
    monkeypatch_target.build_feature_matrix = lambda *_a, **_k: matrix
    try:
        score = _flatbuffer_scorer(
            scorer_model_bin,
            features,
            categorical_encoding,
            signature=signature,
            expected_hash=descriptor.get("features_hash"),
            objective="max",
            score_transform=(descriptor.get("score") or {}).get("transform", "log"),
            physical=True,
        )
        recovered = score(pd.DataFrame(index=range(len(matrix))))
    finally:
        monkeypatch_target.build_feature_matrix = original

    good_score, bad_score = recovered
    assert good_score > bad_score, (
        "the shipped ranker scored a scalar-store instance at least as well as an "
        f"otherwise-identical vectorized one: good={good_score} bad={bad_score}"
    )
