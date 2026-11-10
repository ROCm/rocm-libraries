# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""`linear_data`: the RelLin fit, its artifact, the offline scorer and the train/promote paths."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path

import pytest

pytest.importorskip("flatbuffers")
np = pytest.importorskip("numpy")
pd = pytest.importorskip("pandas")
pytest.importorskip("lightgbm")

from uhd_gen.__main__ import main  # noqa: E402
from uhd_gen.artifact import (  # noqa: E402
    LINEAR_MODEL_IDENTIFIER,
    load_linear_artifact,
    verify_linear_artifact,
)
from uhd_gen.evaluate import _linear_scorer, load_model  # noqa: E402
from uhd_gen.linear_to_flatbuffer import (  # noqa: E402
    build_linear_model,
    write_linear_model,
)
from uhd_gen.rellin import (  # noqa: E402
    IDENTITY,
    LOG1P,
    LinearRanker,
    fit_rellin,
    relative_log_slowdown,
    select_basis,
)

NAN = float("nan")
FEATURES_HASH = "sha256:0123456789abcdef"
CPP_TEST = (
    Path(__file__).resolve().parents[3]
    / "plugin_sdk/tests/heuristics/uhd/TestLinearDataAdapter.cpp"
)


# --------------------------------------------------------------------------------------
# The shared fixture: TestLinearDataAdapter.cpp embeds these bytes and these scores
# --------------------------------------------------------------------------------------


def fixture_ranker() -> LinearRanker:
    """4 features: key (0, 1), family (0), a log1p term over 2 and an identity term over 3."""
    return LinearRanker(
        num_features=4,
        key_features=[0, 1],
        family_features=[0],
        basis_features=np.array([2, 3]),
        transforms=np.array([LOG1P, IDENTITY], dtype=np.uint8),
        mean=np.array([3.0, 0.5]),
        scale=np.array([2.0, 0.25]),
        lo=np.array([-1.5, -2.0]),
        hi=np.array([1.5, 2.0]),
        config_keys=np.array([[1, 1], [1, 2], [2, NAN], [0, 1]], dtype=float),
        config_coefficients=np.array(
            [
                [0.25, 0.5, -0.125],
                [0.0, 0.75, 0.25],
                [0.5, -0.25, 0.375],
                [0.125, 0.0, 0.0],
            ]
        ),
        config_floors=np.array([0.0, 0.0625, 0.0, 0.375]),
        family_keys=np.array([[1], [2]], dtype=float),
        family_coefficients=np.array([[0.1875, 0.25, 0.0625], [0.3125, 0.5, 0.5]]),
        family_floors=np.array([0.0, 0.0]),
        global_coefficients=np.array([0.5, 0.125, 0.25]),
        global_floor=0.0,
    )


FIXTURE_ROWS = np.array(
    [
        [1, 1, 20, 0.75],  # config (1,1)
        [1, 2, 1e9, -50.0],  # config (1,2), both terms clamped
        [2, NAN, NAN, 0.5],  # config (2,NaN); a missing size stands at the mean
        [1, 9, 100, 0.25],  # unseen config, family a=1
        [3, 1, 100, 0.25],  # unseen family, global
        [0, 1, 0, 0.5],  # config (0,1): the raw score is below its floor
        [-0.0, 1, -5, 0.5],  # -0.0 is the key 0.0; a negative size reads as 0
        [2, 7, 20, 2.0],  # unseen config, family a=2
    ]
)


def fixture_bytes() -> bytes:
    return build_linear_model(
        fixture_ranker(),
        FEATURES_HASH,
        100,
        ["gfx942"],
        "1.0.0",
        "2026-01-01T00:00:00+00:00",
    )


def direct_score(ranker: LinearRanker, row) -> float:
    """The documented semantics (linear_model.fbs), one row at a time and without numpy."""
    z = []
    for j in range(ranker.terms):
        x = float(row[ranker.basis_features[j]])
        if math.isnan(x):
            z.append(0.0)
            continue
        t = math.log1p(max(x, 0.0)) if ranker.transforms[j] == LOG1P else x
        z.append(
            min(max((t - ranker.mean[j]) / ranker.scale[j], ranker.lo[j]), ranker.hi[j])
        )

    def same(a, b):
        return (math.isnan(a) and math.isnan(b)) or a == b

    def find(slots, keys):
        wanted = [float(row[slot]) for slot in slots]
        for index, key in enumerate(keys):
            if all(same(a, b) for a, b in zip(wanted, key)):
                return index
        return None

    coefficients, floor = ranker.global_coefficients, ranker.global_floor
    family = (
        find(ranker.family_features, ranker.family_keys)
        if len(ranker.family_keys)
        else None
    )
    if family is not None:
        coefficients, floor = (
            ranker.family_coefficients[family],
            ranker.family_floors[family],
        )
    config = find(ranker.key_features, ranker.config_keys)
    if config is not None:
        coefficients, floor = (
            ranker.config_coefficients[config],
            ranker.config_floors[config],
        )
    total = coefficients[0] + sum(c * v for c, v in zip(coefficients[1:], z))
    return max(total, floor)


def test_the_predictor_is_the_documented_formula():
    ranker = fixture_ranker()
    expected = [direct_score(ranker, row) for row in FIXTURE_ROWS]
    np.testing.assert_allclose(
        ranker.predict(FIXTURE_ROWS), expected, rtol=0, atol=1e-15
    )
    # Spot checks of each rule, by hand (see the C++ test for the same rows).
    assert expected[1] == pytest.approx(0.625)  # both terms at their clamp
    assert expected[2] == 0.5  # NaN size is z = 0, NaN key matches NaN
    assert expected[5] == 0.375  # raised to the config's floor
    assert expected[6] == 0.375  # -0.0 key, negative size


def test_the_cpp_test_embeds_this_fixture_and_these_scores():
    """TestLinearDataAdapter.cpp must test the bytes the writer produces, scored as here."""
    if not CPP_TEST.exists():
        pytest.skip("the C++ tests are not in this checkout")
    text = CPP_TEST.read_text(encoding="utf-8")
    # Whitespace-tolerant: clang-format may break a declaration before its `=`.
    hex_literal = re.search(r"FIXTURE_HEX\s*=\s*((?:\s*\"[0-9a-f]+\"\s*)+);", text)
    digest = re.search(r'FIXTURE_SHA256\s*=\s*"([0-9a-f]{64})"', text)
    scores = re.search(r"FIXTURE_SCORES\s*=\s*\{(.*?)\};", text, re.S)
    assert hex_literal and digest and scores
    embedded = bytes.fromhex(
        "".join(re.findall(r'"([0-9a-f]+)"', hex_literal.group(1)))
    )
    assert embedded == fixture_bytes()
    assert digest.group(1) == hashlib.sha256(embedded).hexdigest()
    recorded = [float(v) for v in re.findall(r"[-0-9.e+]+", scores.group(1))]
    np.testing.assert_allclose(
        recorded, fixture_ranker().predict(FIXTURE_ROWS), rtol=0, atol=1e-15
    )


# --------------------------------------------------------------------------------------
# Writer and verifier
# --------------------------------------------------------------------------------------


def test_the_artifact_round_trips_and_scores_identically(tmp_path):
    ranker = fixture_ranker()
    path = tmp_path / "model.bin"
    digest = write_linear_model(
        ranker,
        FEATURES_HASH,
        path,
        100,
        ["gfx942"],
        "1.0.0",
        "2026-01-01T00:00:00+00:00",
    )
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()
    assert path.read_bytes()[4:8] == LINEAR_MODEL_IDENTIFIER
    loaded = load_linear_artifact(path, digest, feature_count=4)
    assert loaded.key_features == ranker.key_features
    assert loaded.family_features == ranker.family_features
    np.testing.assert_array_equal(
        loaded.predict(FIXTURE_ROWS), ranker.predict(FIXTURE_ROWS)
    )
    assert verify_linear_artifact(path, digest) == path.read_bytes()
    # The writer is a pure function of the model and its stamps.
    assert fixture_bytes() == fixture_bytes()


def _mutated(mutate) -> bytes:
    """The fixture, rewritten through the object API after `mutate(model)`."""
    import flatbuffers
    from hipdnn_flatbuffers_sdk.data_objects.LinearModel import LinearModelT

    model = LinearModelT.InitFromPackedBuf(bytearray(fixture_bytes()), 0)
    mutate(model)
    builder = flatbuffers.Builder(1024)
    builder.Finish(model.Pack(builder), file_identifier=LINEAR_MODEL_IDENTIFIER)
    return bytes(builder.Output())


def _config(model, index):
    return model.configs[index]


REFUSALS = [
    ("negative feature count", lambda m: setattr(m, "numFeatures", -1), "declares -1"),
    (
        "key feature out of range",
        lambda m: setattr(m, "keyFeatures", [0, 4]),
        "reads feature 4",
    ),
    (
        "negative key feature",
        lambda m: setattr(m, "keyFeatures", [-1, 1]),
        "reads feature -1",
    ),
    (
        "family feature out of range",
        lambda m: setattr(m, "familyFeatures", [9]),
        "family_features",
    ),
    (
        "basis feature out of range",
        lambda m: setattr(m.basis[0], "feature", 4),
        "basis term 0 reads feature 4",
    ),
    (
        "unknown transform",
        lambda m: setattr(m.basis[1], "transform", 2),
        "unknown transform 2",
    ),
    (
        "zero scale",
        lambda m: setattr(m.basis[0], "scale", 0.0),
        "scale is not positive",
    ),
    (
        "negative scale",
        lambda m: setattr(m.basis[0], "scale", -1.0),
        "scale is not positive",
    ),
    ("lo above hi", lambda m: setattr(m.basis[0], "lo", 2.0), "lo is above its hi"),
    ("infinite hi", lambda m: setattr(m.basis[0], "hi", math.inf), "not finite"),
    ("nan mean", lambda m: setattr(m.basis[0], "mean", NAN), "not finite"),
    (
        "short coefficients",
        lambda m: setattr(_config(m, 0), "coefficients", [1.0, 2.0]),
        "config 0 has 2 coefficients, expected 3",
    ),
    (
        "nonfinite coefficient",
        lambda m: setattr(_config(m, 1), "coefficients", [0.0, math.inf, 0.0]),
        "config 1's coefficient 1 is not finite",
    ),
    (
        "nonfinite floor",
        lambda m: setattr(_config(m, 0), "floor", NAN),
        "config 0's floor is not finite",
    ),
    (
        "short key",
        lambda m: setattr(_config(m, 2), "key", [1.0]),
        "config 2's key has 1 values, expected 2",
    ),
    (
        "duplicate key",
        lambda m: setattr(_config(m, 1), "key", [1.0, 1.0]),
        "config 1 repeats an earlier config's key",
    ),
    (
        "duplicate zero key",
        lambda m: (
            setattr(_config(m, 0), "key", [0.0, 5.0]),
            setattr(_config(m, 1), "key", [-0.0, 5.0]),
        ),
        "config 1 repeats an earlier config's key",
    ),
    (
        "duplicate nan key",
        lambda m: (
            setattr(_config(m, 0), "key", [NAN, 5.0]),
            setattr(_config(m, 1), "key", [-NAN, 5.0]),
        ),
        "config 1 repeats an earlier config's key",
    ),
    (
        "families without a family key",
        lambda m: setattr(m, "familyFeatures", []),
        "no family_features",
    ),
    (
        "duplicate family key",
        lambda m: setattr(m.families[1], "key", [1.0]),
        "family 1 repeats an earlier family's key",
    ),
    (
        "no global entry",
        lambda m: setattr(m, "global_", None),
        "global entry is missing",
    ),
    (
        "global with a key",
        lambda m: setattr(m.global_, "key", [1.0]),
        "the global entry's key has 1 values, expected 0",
    ),
]


@pytest.mark.parametrize(
    "mutate, reason", [pytest.param(m, r, id=n) for n, m, r in REFUSALS]
)
def test_an_inconsistent_model_is_refused_for_the_reason_the_runtime_gives(
    tmp_path, mutate, reason
):
    path = tmp_path / "model.bin"
    path.write_bytes(_mutated(mutate))
    with pytest.raises(ValueError, match=re.escape(reason)):
        load_linear_artifact(path, None)


def test_a_buffer_the_runtime_would_not_open_is_refused_in_its_order(tmp_path):
    path = tmp_path / "model.bin"
    good = fixture_bytes()

    path.write_bytes(good)
    with pytest.raises(ValueError, match="model hash mismatch"):
        load_linear_artifact(path, "0" * 64)
    with pytest.raises(ValueError, match="num_features 4 differs"):
        load_linear_artifact(path, None, feature_count=5)

    path.write_bytes(good[:4] + b"TABL" + good[8:])
    with pytest.raises(ValueError, match="file identifier"):
        load_linear_artifact(path, None)

    path.write_bytes(good[:4] + b"HGBM" + good[8:])
    with pytest.raises(ValueError, match="file identifier"):
        load_linear_artifact(path, None)

    path.write_bytes(good[: len(good) // 2])
    with pytest.raises(ValueError, match="verifier refuses"):
        load_linear_artifact(path, None)

    path.write_bytes(b"")
    with pytest.raises(ValueError, match="artifact size"):
        load_linear_artifact(path, None)

    # A root offset pointing past the end of the buffer.
    corrupt = bytearray(good)
    corrupt[0:4] = (len(good) + 100).to_bytes(4, "little")
    path.write_bytes(bytes(corrupt))
    with pytest.raises(ValueError, match="verifier refuses"):
        load_linear_artifact(path, None)


# --------------------------------------------------------------------------------------
# The fit
# --------------------------------------------------------------------------------------


def test_the_target_is_the_log_slowdown_in_the_objectives_direction():
    problems = np.array([0, 0, 0, 1, 1])
    latency = np.array([2.0, 4.0, 3.0, 5.0, 5.0])
    np.testing.assert_allclose(
        relative_log_slowdown(latency, problems, "min"),
        np.log([1.0, 2.0, 1.5, 1.0, 1.0]),
    )
    rate = np.array([100.0, 50.0, 25.0, 8.0, 4.0])
    np.testing.assert_allclose(
        relative_log_slowdown(rate, problems, "max"),
        np.log([1.0, 2.0, 4.0, 1.0, 2.0]),
    )
    with pytest.raises(ValueError, match="finite and positive"):
        relative_log_slowdown(np.array([1.0, 0.0]), np.array([0, 0]), "min")
    with pytest.raises(ValueError, match="objective"):
        relative_log_slowdown(latency, problems, "mean")


def test_the_basis_is_every_varying_non_key_slot_logged_when_it_spans_orders():
    matrix = np.column_stack(
        [
            np.tile([1.0, 2.0, 3.0], 20),  # key
            np.full(60, 7.0),  # constant
            np.linspace(0, 10, 60),  # small span
            np.geomspace(1, 4096, 60),  # large span, non-negative
            np.linspace(-500, 500, 60),  # large span but negative
            np.where(np.arange(60) % 2, NAN, 5.0),  # constant where present
        ]
    )
    slots, transforms = select_basis(matrix, [0])
    assert slots == [2, 3, 4]
    assert transforms == [IDENTITY, LOG1P, IDENTITY]


def _corpus(seed=0, problems=240):
    """Configs 0-3 over problem size s: each is fastest in its own band of sizes."""
    rng = np.random.default_rng(seed)
    rows = []
    for problem in range(problems):
        size = float(np.exp(rng.uniform(0, math.log(4096))))
        for config, (slope, offset) in enumerate(
            [(0.2, 0.0), (0.5, -1.2), (0.8, -3.0), (1.1, -5.6)]
        ):
            latency = math.exp(slope * math.log(size) + offset + rng.normal(0, 0.01))
            rows.append((problem, config, size, 2.0 * size, latency))
    frame = pd.DataFrame(
        rows, columns=["problem", "config", "size", "size2", "latency"]
    )
    return frame


def _fit(frame, **kwargs):
    matrix = frame[["config", "size", "size2"]].to_numpy(dtype=float)
    y = relative_log_slowdown(
        frame["latency"].to_numpy(), frame["problem"].to_numpy(), "min"
    )
    basis, transforms = select_basis(matrix, [0])
    return (
        fit_rellin(
            matrix,
            y,
            key_features=[0],
            basis_features=basis,
            transforms=transforms,
            **kwargs,
        ),
        matrix,
        y,
    )


def _regret(frame, scores):
    picked = frame.assign(score=scores).sort_values("score").groupby("problem").head(1)
    best = frame.groupby("problem")["latency"].min()
    return float(
        np.mean(
            picked.set_index("problem")["latency"]
            / best.loc[picked["problem"]].to_numpy()
            - 1
        )
    )


def test_the_fit_ranks_the_planted_best_config_on_problems_it_never_saw():
    frame = _corpus()
    train = frame[frame["problem"] < 160]
    held = frame[frame["problem"] >= 160]
    ranker, _, _ = _fit(train)
    regret = _regret(
        held, ranker.predict(held[["config", "size", "size2"]].to_numpy(dtype=float))
    )
    random_regret = _regret(held, np.random.default_rng(1).random(len(held)))
    assert regret < 0.02
    assert random_regret > 10 * regret


def test_an_infinite_training_value_still_writes_a_model_the_runtime_loads(tmp_path):
    frame = _corpus(problems=60)
    clean, _, _ = _fit(frame)
    frame.loc[3, "size2"] = math.inf
    frame.loc[7, "size2"] = -math.inf
    ranker, matrix, _ = _fit(frame)
    for values in (ranker.mean, ranker.scale, ranker.lo, ranker.hi):
        assert np.isfinite(values).all()
    # The loader's checks (non-finite mean, scale, lo or hi is refused) pass.
    path = tmp_path / "model.bin"
    digest = write_linear_model(ranker, FEATURES_HASH, path)
    load_linear_artifact(path, digest, feature_count=3)
    # An infinite value is clamped to the range training saw, as the runtime clamps it.
    assert np.isfinite(ranker.predict(matrix)).all()
    assert np.isfinite(ranker.predict(np.array([[0.0, 10.0, math.inf]]))).all()
    # The finite rows set the standardization the clean corpus would have had.
    term = list(ranker.basis_features).index(2)
    assert ranker.mean[term] == pytest.approx(clean.mean[term], rel=0.05)


def test_a_term_is_clamped_to_its_training_range_and_a_floor_is_each_configs_best():
    frame = _corpus()
    ranker, matrix, y = _fit(frame[frame["size"] < 500])
    inside = frame[frame["size"] < 500]
    # A config's floor is the smallest slowdown it showed in training.
    for config, floor in zip(ranker.config_keys[:, 0], ranker.config_floors):
        observed = relative_log_slowdown(
            inside["latency"].to_numpy(), inside["problem"].to_numpy(), "min"
        )[(inside["config"] == config).to_numpy()]
        assert floor == pytest.approx(observed.min())
    # A config that is never the best has a floor above zero.
    assert ranker.config_floors.max() > 0.1

    edge = np.array(
        [[c, inside["size"].max(), 2 * inside["size"].max()] for c in range(4)]
    )
    far = np.array([[c, 1e12, 2e12] for c in range(4)])
    np.testing.assert_allclose(ranker.predict(far), ranker.predict(edge), atol=1e-12)


def test_the_floor_stops_a_config_from_being_extrapolated_into_a_winner():
    ranker = fixture_ranker()
    ranker.config_coefficients[0] = [-5.0, 0.0, 0.0]  # raw -5 everywhere
    ranker.config_floors[0] = 0.2
    row = np.array([[1, 1, 20, 0.75]])
    assert ranker.predict(row)[0] == 0.2
    # Without the floor the config would have beaten every other.
    ranker.config_floors[0] = -10.0
    ranker._indices.clear()
    assert ranker.predict(row)[0] == -5.0


def test_a_config_with_few_rows_is_shrunk_toward_the_pooled_fit():
    frame = _corpus()
    thin = pd.concat([frame[frame["config"] != 3], frame[frame["config"] == 3].head(3)])
    loose, _, _ = _fit(thin, min_rows=0)
    tight, _, _ = _fit(thin, min_rows=10)
    index = list(tight.config_keys[:, 0]).index(3)
    spread = lambda r: np.abs(
        r.config_coefficients[index] - r.global_coefficients
    ).sum()  # noqa: E731
    assert spread(tight) < spread(loose)


def test_a_family_serves_a_config_the_corpus_never_measured():
    frame = _corpus()
    frame["family"] = frame["config"] // 2
    matrix = frame[["config", "family", "size"]].to_numpy(dtype=float)
    y = relative_log_slowdown(
        frame["latency"].to_numpy(), frame["problem"].to_numpy(), "min"
    )
    seen = frame["config"] != 3
    basis, transforms = select_basis(matrix, [0, 1])
    ranker = fit_rellin(
        matrix[seen.to_numpy()],
        y[seen.to_numpy()],
        key_features=[0],
        family_features=[1],
        basis_features=basis,
        transforms=transforms,
    )
    assert len(ranker.family_keys) == 2
    unseen = np.array([[3, 1, 100.0]])
    family_one = ranker.family_coefficients[list(ranker.family_keys[:, 0]).index(1)]
    z = ranker.basis(unseen)[0]
    assert ranker.predict(unseen)[0] == pytest.approx(
        max(family_one[0] + family_one[1:] @ z, 0.0)
    )
    # With no family key every unseen config falls back to the global fit.
    flat, _, _ = _fit(frame[seen])
    assert len(flat.family_keys) == 0
    z = flat.basis(np.array([[3, 100.0, 200.0]]))[0]
    expected = flat.global_coefficients[0] + flat.global_coefficients[1:] @ z
    assert flat.predict(np.array([[3, 100.0, 200.0]]))[0] == pytest.approx(
        max(expected, 0.0)
    )


# --------------------------------------------------------------------------------------
# The offline scorer
# --------------------------------------------------------------------------------------


def test_the_offline_scorer_is_the_direct_evaluation_in_the_corpus_direction(tmp_path):
    ranker = fixture_ranker()
    path = tmp_path / "model.bin"
    digest = write_linear_model(ranker, FEATURES_HASH, path)
    features = ["f0", "f1", "f2", "f3"]
    # The shared feature path refuses non-finite values, so only the finite rows go through it.
    rows = FIXTURE_ROWS[~np.isnan(FIXTURE_ROWS).any(axis=1)]
    frame = pd.DataFrame(rows, columns=features)
    expected = np.array([direct_score(ranker, row) for row in rows])

    lower = _linear_scorer(path, features, model_hash=digest, objective="min")
    np.testing.assert_allclose(lower(frame), expected, rtol=0, atol=1e-15)
    higher = _linear_scorer(path, features, model_hash=digest, objective="max")
    np.testing.assert_allclose(higher(frame), -expected, rtol=0, atol=1e-15)

    with pytest.raises(ValueError, match="oriented by the corpus"):
        _linear_scorer(path, features, objective=None)
    with pytest.raises(ValueError, match="model hash mismatch"):
        _linear_scorer(path, features, model_hash="0" * 64, objective="min")
    with pytest.raises(ValueError, match="does not match the shipped model"):
        _linear_scorer(
            path, features, expected_hash="sha256:ffffffffffffffff", objective="min"
        )


# --------------------------------------------------------------------------------------
# train, evaluate and promote
# --------------------------------------------------------------------------------------

PROVENANCE = {
    "ued": {"id": "13ab344f-4818-4772-bb8e-8e1441fec82c", "revision": "1.0"},
    "kmd": {"id": "46d64d06-18eb-483d-9bb4-94472d32b78d", "revision": "1.0"},
    "umd": [],
}


def _catalog(problems=60):
    """A catalog corpus: 4 configs (two kernel knobs) over `problems` sizes, in TFLOPS."""
    rng = np.random.default_rng(3)
    rows = []
    for problem in range(problems):
        size = float(np.exp(rng.uniform(0, math.log(4096))))
        for tile, stages in ((64, 2), (64, 4), (128, 2), (128, 4)):
            time = math.exp(
                0.3 * math.log(size) * (1 if tile == 64 else 0.6) + stages * 0.1
            )
            rows.append(
                {
                    "benchmark": f"p{problem}",
                    "device": "board",
                    "kernel.tile": tile,
                    "kernel.stages": stages,
                    "q.size": size,
                    "q.size2": size * 2 + 1,
                    "tflops": 1000.0 / time,
                }
            )
    return pd.DataFrame(rows)


def _train(tmp_path, evaluator, *extra, frame=None):
    corpus = tmp_path / "corpus.csv"
    (frame if frame is not None else _catalog()).to_csv(corpus, index=False)
    snapshot = tmp_path / "provenance.json"
    snapshot.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    output = tmp_path / "model"
    code = main(
        [
            "train",
            "--adapter",
            "linear_data",
            "--input",
            str(corpus),
            "--provenance",
            str(snapshot),
            "--features",
            "kernel.tile",
            "kernel.stages",
            "q.size",
            "q.size2",
            "--group-by",
            "benchmark",
            "device",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(output),
            "--training-arches",
            "gfx942",
            *extra,
        ]
    )
    return code, output


def test_train_emits_a_valid_linear_data_uhd(tmp_path, evaluator):
    code, output = _train(tmp_path, evaluator)
    assert code == 0
    descriptor = json.loads((output / "heuristic.uhd.json").read_text(encoding="utf-8"))
    assert descriptor["adapter"] == "linear_data"
    assert descriptor["objective"] == "min"
    assert "score" not in descriptor and "tree_data" not in descriptor
    body = descriptor["linear_data"]
    assert body["artifact"] == "model.bin"
    artifact = output / "model.bin"
    assert body["hash"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    # The runtime's checks, in its order, against the descriptor's own signature.
    ranker = load_linear_artifact(
        artifact, body["hash"], feature_count=len(descriptor["features_signature"])
    )
    assert ranker.key_features == [0, 1]  # the `$kernel.*` references
    assert len(ranker.config_keys) == 4
    assert [int(slot) for slot in ranker.basis_features] == [2, 3]
    manifest = json.loads((output / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["adapter"] == "linear_data"
    assert (
        manifest["objective"] == "max"
    )  # the corpus's TFLOPS, not the score's direction
    assert manifest["key_features"] == ["kernel.tile", "kernel.stages"]
    assert manifest["score_metric"] is None and manifest["score_calibrated"] is False
    assert not (output / "model.lgbm").exists()


def test_the_trained_model_ranks_the_fastest_candidate_first_when_evaluated(
    tmp_path, evaluator
):
    frame = _catalog()
    code, output = _train(tmp_path, evaluator, frame=frame)
    assert code == 0
    bundle = load_model(output, feature_evaluator=evaluator)
    assert bundle.objective == "max"  # the measured target's direction, for the report
    assert bundle.descriptor["objective"] == "min"
    scores = bundle.scorer(frame)
    # Higher is better in the bundle's objective, as for any scorer.
    best = frame.assign(score=scores).sort_values("score").groupby("benchmark").tail(1)
    oracle = (
        frame.sort_values("tflops").groupby("benchmark").tail(1).set_index("benchmark")
    )
    assert (best.set_index("benchmark")["tflops"] / oracle["tflops"]).min() > 0.99
    # `--objective` overrides the direction of the corpus the score is compared with.
    flipped = load_model(output, feature_evaluator=evaluator, objective_override="min")
    np.testing.assert_allclose(flipped.scorer(frame), -scores, atol=1e-15)


def test_an_installed_model_without_a_manifest_needs_the_corpus_direction(
    tmp_path, evaluator
):
    frame = _catalog()
    code, output = _train(tmp_path, evaluator, frame=frame)
    assert code == 0
    # As promote installs it: the descriptor and model.bin, no train_manifest.json.
    (output / "train_manifest.json").unlink()
    # The descriptor's `min` orients the runtime score, not the TFLOPS the report ranks.
    with pytest.raises(ValueError, match="pass --objective"):
        load_model(output, feature_evaluator=evaluator)
    bundle = load_model(output, feature_evaluator=evaluator, objective_override="max")
    best = frame.assign(score=bundle.scorer(frame)).sort_values("score")
    best = best.groupby("benchmark").tail(1).set_index("benchmark")
    oracle = (
        frame.sort_values("tflops").groupby("benchmark").tail(1).set_index("benchmark")
    )
    assert (best["tflops"] / oracle["tflops"]).min() > 0.99


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--metric", "tflops"], "drop --metric"),
        (["--calibrated"], "drop --metric"),
        (["--group-by-feature", "kernel.tile"], "tree_data option"),
        (["--key-features", "kernel.nonexistent"], "not among the model's features"),
    ],
)
def test_train_refuses_what_a_slowdown_ranker_cannot_be(
    tmp_path, evaluator, caplog, extra, message
):
    code, output = _train(tmp_path, evaluator, *extra)
    assert code == 1
    assert message in caplog.text
    assert not output.exists()


def test_train_needs_problem_identity_and_a_key(tmp_path, evaluator, caplog):
    corpus = tmp_path / "corpus.csv"
    _catalog().drop(columns=["benchmark"]).to_csv(corpus, index=False)
    snapshot = tmp_path / "provenance.json"
    snapshot.write_text(json.dumps(PROVENANCE), encoding="utf-8")
    code = main(
        [
            "train",
            "--adapter",
            "linear_data",
            "--input",
            str(corpus),
            "--provenance",
            str(snapshot),
            "--features",
            "kernel.tile",
            "q.size",
            "--feature-evaluator",
            evaluator,
            "--output-dir",
            str(tmp_path / "model"),
        ]
    )
    assert code == 1
    assert "needs problem identity" in caplog.text

    # An explicit key replaces the `$kernel.*` default.
    (tmp_path / "other").mkdir()
    code, _ = _train(tmp_path / "other", evaluator, "--key-features", "kernel.tile")
    assert code == 0


def test_promote_installs_a_linear_data_uhd_the_runtime_would_load(tmp_path, evaluator):
    from uhd_gen.promote import PromoteError, _apply, build_plan
    from uhd_gen.provenance import snapshot_provenance
    from uhd_gen.tests.test_promote import (
        KERNEL_SIGNATURE,
        NEW,
        _hash,
        _read,
        _tree,
        _write,
    )

    tree = _tree(tmp_path / "tree")

    def model(
        root, *, leaf, features=1, signature=KERNEL_SIGNATURE, features_hash=None
    ):
        ranker = LinearRanker(
            num_features=features,
            key_features=[0],
            family_features=[],
            basis_features=np.array([], dtype=np.int64),
            transforms=np.array([], dtype=np.uint8),
            mean=np.zeros(0),
            scale=np.zeros(0),
            lo=np.zeros(0),
            hi=np.zeros(0),
            config_keys=np.array([[64.0], [128.0]]),
            config_coefficients=np.array([[0.0], [leaf]]),
            config_floors=np.array([0.0, 0.0]),
            family_keys=np.empty((0, 0)),
            family_coefficients=np.empty((0, 1)),
            family_floors=np.empty(0),
            global_coefficients=np.array([1.0]),
        )
        root.mkdir(parents=True, exist_ok=True)
        digest = write_linear_model(
            ranker,
            features_hash or _hash(tuple(signature)),
            root / "model.bin",
            training_arches=["gfx942"],
        )
        provenance = snapshot_provenance(tree, arch="gfx942")
        _write(
            root / "heuristic.uhd.json",
            {
                "version": "1.0",
                "id": NEW,
                "name": "linear",
                "adapter": "linear_data",
                "objective": "min",
                "features_signature": list(signature),
                "features_hash": _hash(tuple(signature)),
                "trained_against": provenance,
                "linear_data": {"artifact": "model.bin", "hash": digest},
            },
        )
        _write(
            root / "train_manifest.json",
            {"training_arches": ["gfx942"], "trained_against": provenance},
        )
        return root

    first = model(tmp_path / "first", leaf=0.5)
    plan = build_plan(first, tree)
    _apply(plan)
    installed = plan.destination_descriptor
    assert _read(installed)["adapter"] == "linear_data"
    assert _read(tree / "engine.ued.json")["sort_kernel_catalog"]["gfx942"] == [NEW]

    # The same id retrained is written over the model at its own layout, as for any adapter.
    second = model(tmp_path / "second", leaf=0.25)
    again = build_plan(second, tree)
    assert again.destination_descriptor == installed

    # A linear model has no calibrated metric, so it can never serve predict_engine.
    with pytest.raises(PromoteError, match="ranking metric|metric"):
        build_plan(first, tree, "ASM_SDPA_ENGINE", role="predict_engine", arch="gfx942")

    # Whatever the runtime would refuse is refused before a write.
    wide = model(tmp_path / "wide", leaf=0.5, features=2)
    with pytest.raises(PromoteError, match="engine would refuse.*num_features 2"):
        build_plan(wide, tree)
    stale = model(tmp_path / "stale", leaf=0.5, features_hash="sha256:" + "f" * 16)
    with pytest.raises(PromoteError, match="features_hash"):
        build_plan(stale, tree)

    document = _read(first / "heuristic.uhd.json")
    for edit, message in (
        ({"objective": "max"}, "linear_data requires objective min"),
        ({"score": {"metric": "time"}}, "linear_data requires objective min"),
        ({"score": {"transform": "log1p"}}, "linear_data requires objective min"),
    ):
        broken = first.parent / "broken"
        broken.mkdir(exist_ok=True)
        _write(broken / "heuristic.uhd.json", {**document, **edit})
        _write(broken / "train_manifest.json", _read(first / "train_manifest.json"))
        (broken / "model.bin").write_bytes((first / "model.bin").read_bytes())
        with pytest.raises(PromoteError, match=message):
            build_plan(broken, tree)
