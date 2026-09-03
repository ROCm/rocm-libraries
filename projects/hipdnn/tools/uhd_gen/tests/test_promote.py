#!/usr/bin/env python3
# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT
"""The last manual step in the pipeline, and the ways it used to go wrong.

A trained UHD only takes effect once the engine's UED names its id in `heuristic`.
When that hand-edit was skipped, wrong, or half-applied, nothing failed loudly: the
engine either kept ranking by priority or was dropped by the loader, and the only
symptom was that the model "did nothing". Every test here pins one of those silences
shut -- an id that is not the model's, a guessed engine, a descriptor installed
without its artifact, a dry run that wrote anyway.
"""
from __future__ import annotations

import json
import uuid
from pathlib import Path

import pytest

# Optional heavyweight training dependencies. Imported before uhd_gen.__main__, which
# pulls them in transitively, so a missing dep is a skip rather than a collection error.
pytest.importorskip("lightgbm")
pytest.importorskip("pandas")
pytest.importorskip("flatbuffers")

import uhd_gen  # noqa: E402,F401  puts _generated/ on sys.path
from uhd_gen.__main__ import main  # noqa: E402

#: A UED as the loader expects one (RFC 0020 §4.2, mirrored by the packaged
#: pointwise_model fixture): schema-less, version-gated, heuristic by id.
UED_ID_A = "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d62"
UED_ID_B = "1a4f7c30-0d92-4c1b-8a6e-2f5b9d3e7c08"
STALE_HEURISTIC = "727e5401-3b99-49ff-a2fc-68fd4eedbb54"
METADATA_ID = "3f8a1c07-52d9-4e61-b0a4-9c7d61e2830f"


def _write_json(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2)
        handle.write("\n")


def _make_model_dir(
    root: Path,
    uhd_id: str,
    *,
    stem: str = "heuristic",
    artifact: str = "model.bin",
    write_artifact: bool = True,
) -> Path:
    """A stand-in for a `uhd_gen train --output-dir` result.

    Hand-built rather than trained: promote only reads the descriptor and copies the
    artifact bytes, so training here would test LightGBM, slowly.
    """
    root.mkdir(parents=True, exist_ok=True)
    _write_json(
        root / f"{stem}.uhd.json",
        {
            "version": "1.0",
            "id": uhd_id,
            "name": "test selector",
            "adapter": "tree_data",
            "features_signature": ["$kernel.block_size"],
            "features_hash": "sha256:bc673de29ad2cc2c",
            "objective": "max",
            "score": {"units": "tflops", "calibrated": False, "transform": "log1p"},
            "tree_data": {"artifact": artifact},
        },
    )
    if write_artifact:
        (root / artifact).write_bytes(b"HDNN-model-bytes-" + uhd_id.encode("ascii"))
    return root


def _make_ued(
    path: Path, name: str, ued_id: str, heuristic: str | None = STALE_HEURISTIC
) -> Path:
    document = {"version": "1.0", "id": ued_id, "name": name}
    if heuristic is not None:
        # Placed before `metadata`, not last: an updater that appends the key instead
        # of replacing it in place would move it, and the diff would stop being one
        # reviewable line.
        document["heuristic"] = heuristic
    document["metadata"] = METADATA_ID
    document["knobs"] = ["block_size"]
    _write_json(path, document)
    return path


def _heuristic_of(path: Path) -> str | None:
    return json.loads(path.read_text(encoding="utf-8")).get("heuristic")


def _training_csv(path: Path) -> Path:
    """The smallest CSV that trains: one kernel knob, a target that varies with it.

    Shaped like the packaged pointwise fixture (kernel.block_size -> tflops); enough
    rows for the 5-fold CV in train_model.
    """
    rows = ["kernel.block_size,tflops"]
    for index in range(40):
        rows.append(f"64,{90.0 + index * 0.01:.2f}")
        rows.append(f"256,{50.0 + index * 0.01:.2f}")
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _train(output_dir: Path, csv: Path, *extra: str) -> int:
    return main(
        [
            "train",
            "--input",
            str(csv),
            "--features",
            "kernel.block_size",
            "--target",
            "tflops",
            "--output-dir",
            str(output_dir),
            "--num-boost-round",
            "10",
            "--early-stopping",
            "5",
            *extra,
        ]
    )


# --------------------------------------------------------------------------------
# Piece 1: train --uhd-id
# --------------------------------------------------------------------------------


def test_train_uhd_id_is_the_descriptor_identity(tmp_path):
    """The point of --uhd-id: retraining keeps the id the UED already names.

    If train minted a fresh id anyway, the UED would still point at the previous
    descriptor -- which no longer exists -- and the engine would load with no
    heuristic at all.
    """
    requested = str(uuid.uuid4())
    output_dir = tmp_path / "model"

    assert _train(output_dir, _training_csv(tmp_path / "bench.csv"), "--uhd-id", requested) == 0

    descriptor = json.loads((output_dir / "heuristic.uhd.json").read_text(encoding="utf-8"))
    assert descriptor["id"] == requested
    manifest = json.loads((output_dir / "train_manifest.json").read_text(encoding="utf-8"))
    assert manifest["uhd_id"] == requested


def test_train_without_uhd_id_still_mints_one(tmp_path):
    """The flag is optional; omitting it must keep the previous behaviour."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _training_csv(tmp_path / "bench.csv")) == 0

    descriptor = json.loads((output_dir / "heuristic.uhd.json").read_text(encoding="utf-8"))
    uuid.UUID(descriptor["id"])  # raises if it is not a UUID


@pytest.mark.parametrize(
    "malformed",
    [
        "not-a-uuid",
        "",
        # One hex digit short: the kind of typo a copy-paste produces, and the kind the
        # loader answers with a silently heuristic-less engine.
        "6d2b90f4-8c15-4a37-9e58-04b7c3fa1d6",
    ],
)
def test_train_rejects_a_malformed_uhd_id(tmp_path, malformed):
    """A bad id must fail the run, not become the descriptor's identity."""
    output_dir = tmp_path / "model"

    assert _train(output_dir, _training_csv(tmp_path / "bench.csv"), "--uhd-id", malformed) == 1
    # Rejected before any output exists: the check runs ahead of training, so a typo
    # costs an error message rather than a training run.
    assert not (output_dir / "heuristic.uhd.json").exists()


# --------------------------------------------------------------------------------
# Piece 2: promote
# --------------------------------------------------------------------------------


def test_promote_points_the_ued_at_the_model_and_installs_the_pair(tmp_path, capsys):
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A)

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 0

    assert _heuristic_of(ued) == uhd_id
    assert (tree / "heuristic.uhd.json").is_file()
    assert (tree / "model.bin").read_bytes() == (model_dir / "model.bin").read_bytes()
    # The installed descriptor is the model's, not a copy carrying some other id.
    installed = json.loads((tree / "heuristic.uhd.json").read_text(encoding="utf-8"))
    assert installed["id"] == uhd_id

    report = capsys.readouterr().out
    assert STALE_HEURISTIC in report and uhd_id in report


def test_promote_preserves_ued_formatting_and_key_order(tmp_path):
    """One changed line, not a reformat: a promote nobody can review is a promote
    nobody applies."""
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A)
    before = ued.read_text(encoding="utf-8").splitlines()

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 0

    after = ued.read_text(encoding="utf-8").splitlines()
    assert len(before) == len(after)
    differing = [index for index, (a, b) in enumerate(zip(before, after)) if a != b]
    assert differing == [4], f"expected only the heuristic line to change, got {differing}"
    assert ued.read_text(encoding="utf-8").endswith("}\n")


def test_promote_adds_heuristic_to_a_ued_that_had_none(tmp_path, capsys):
    """An engine that ranked by priority is the common starting state."""
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A, heuristic=None)

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 0

    assert _heuristic_of(ued) == uhd_id
    assert "(none)" in capsys.readouterr().out


def test_promote_refuses_to_guess_between_two_ueds(tmp_path):
    """Refusing beats choosing.

    Promoting into the wrong engine fails twice over: the engine that was retrained
    keeps its old model, and an unrelated engine starts ranking with a model trained
    for a different kernel set. Both load cleanly and report nothing.
    """
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)
    tree = tmp_path / "tree"
    first = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A)
    second = _make_ued(tree / "b.ued.json", "hipkernel:reduction", UED_ID_B)

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 1

    assert _heuristic_of(first) == STALE_HEURISTIC
    assert _heuristic_of(second) == STALE_HEURISTIC
    assert not (tree / "heuristic.uhd.json").exists()
    assert not (tree / "model.bin").exists()


def test_promote_engine_selects_the_named_ued_only(tmp_path):
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)
    tree = tmp_path / "tree"
    first = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A)
    second = _make_ued(tree / "nested" / "b.ued.json", "hipkernel:reduction", UED_ID_B)

    assert (
        main(
            [
                "promote",
                "--model-dir",
                str(model_dir),
                "--descriptor-tree",
                str(tree),
                "--engine",
                "hipkernel:reduction",
            ]
        )
        == 0
    )

    assert _heuristic_of(second) == uhd_id
    assert _heuristic_of(first) == STALE_HEURISTIC
    # The pair lands beside the UED that was updated, which is where the loader
    # resolves `tree_data.artifact` from.
    assert (second.parent / "heuristic.uhd.json").is_file()
    assert (second.parent / "model.bin").is_file()
    assert not (first.parent / "model.bin").exists()


def test_promote_rejects_an_unknown_engine_name(tmp_path):
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A)

    assert (
        main(
            [
                "promote",
                "--model-dir",
                str(model_dir),
                "--descriptor-tree",
                str(tree),
                "--engine",
                "hipkernel:typo",
            ]
        )
        == 1
    )
    assert _heuristic_of(ued) == STALE_HEURISTIC


def test_promote_dry_run_writes_nothing(tmp_path, capsys):
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A)
    before = ued.read_text(encoding="utf-8")

    assert (
        main(
            [
                "promote",
                "--model-dir",
                str(model_dir),
                "--descriptor-tree",
                str(tree),
                "--dry-run",
            ]
        )
        == 0
    )

    assert ued.read_text(encoding="utf-8") == before
    assert sorted(path.name for path in tree.iterdir()) == ["engine.ued.json"]
    output = capsys.readouterr().out
    assert "dry run" in output and uhd_id in output


def test_promote_refuses_a_model_dir_missing_its_artifact(tmp_path):
    """Installing a descriptor whose artifact is absent is the worst outcome available:
    the loader finds the UHD, fails to build the adapter, and drops the engine."""
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id, write_artifact=False)
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A)

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 1

    assert _heuristic_of(ued) == STALE_HEURISTIC
    assert not (tree / "heuristic.uhd.json").exists()


def test_promote_refuses_a_descriptor_with_a_non_uuid_id(tmp_path):
    model_dir = _make_model_dir(tmp_path / "model", "definitely-not-a-uuid")
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A)

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 1

    assert _heuristic_of(ued) == STALE_HEURISTIC
    assert not (tree / "model.bin").exists()


def test_promote_refuses_an_empty_model_dir(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A)

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 1
    assert _heuristic_of(ued) == STALE_HEURISTIC


def test_promote_refuses_a_tree_with_no_ued(tmp_path):
    model_dir = _make_model_dir(tmp_path / "model", str(uuid.uuid4()))
    tree = tmp_path / "tree"
    tree.mkdir()

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 1
    assert not (tree / "model.bin").exists()


def test_promote_refuses_to_strand_another_engines_heuristic(tmp_path):
    """Same filename, different id, and someone else points at the old one.

    Overwriting would leave the other UED naming an id nothing defines, which the
    loader answers by dropping that engine entirely.
    """
    tree = tmp_path / "tree"
    incumbent = _make_model_dir(tree, str(uuid.uuid4()))
    incumbent_id = json.loads(
        (tree / "heuristic.uhd.json").read_text(encoding="utf-8")
    )["id"]
    target = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A)
    other = _make_ued(tree / "b.ued.json", "hipkernel:reduction", UED_ID_B, incumbent_id)

    model_dir = _make_model_dir(tmp_path / "model", str(uuid.uuid4()))

    assert (
        main(
            [
                "promote",
                "--model-dir",
                str(model_dir),
                "--descriptor-tree",
                str(tree),
                "--engine",
                "hipkernel:pointwise",
            ]
        )
        == 1
    )

    assert _heuristic_of(target) == STALE_HEURISTIC
    assert _heuristic_of(other) == incumbent_id
    assert json.loads((tree / "heuristic.uhd.json").read_text(encoding="utf-8"))[
        "id"
    ] == incumbent_id


def test_promote_refuses_to_clobber_a_neighbours_artifact(tmp_path):
    """`train` calls every artifact `model.bin`, so two stems collide by default.

    The victim descriptor keeps its features_hash and its filename, and only its trees
    change -- a mismatch that shows up as different rankings, not as an error.
    """
    tree = tmp_path / "tree"
    _make_model_dir(tree, str(uuid.uuid4()), stem="packed_pointwise_model")
    ued = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A)
    model_dir = _make_model_dir(tmp_path / "model", str(uuid.uuid4()), stem="heuristic")

    assert main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 1

    assert _heuristic_of(ued) == STALE_HEURISTIC
    assert (tree / "model.bin").read_bytes() != (model_dir / "model.bin").read_bytes()


def test_promote_warns_when_replacing_an_unreferenced_descriptor(tmp_path, caplog):
    """Nothing is stranded, so this proceeds -- but silently swapping a file that is a
    different heuristic, not a newer build of the same one, is worth saying out loud."""
    tree = tmp_path / "tree"
    _make_model_dir(tree, str(uuid.uuid4()))
    ued = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A)
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)

    with caplog.at_level("WARNING"):
        assert (
            main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 0
        )

    assert "OVERWRITING" in caplog.text
    assert _heuristic_of(ued) == uhd_id


def test_promote_does_not_warn_when_superseding_the_engines_own_heuristic(tmp_path, caplog):
    """The ordinary retrain must stay quiet.

    The descriptor being replaced is the one this engine already ranks by, so the
    replacement is the whole point. Warning here would train readers to scroll past the
    warning that means something -- an unrelated heuristic being clobbered.
    """
    tree = tmp_path / "tree"
    incumbent_id = str(uuid.uuid4())
    _make_model_dir(tree, incumbent_id)
    ued = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A, incumbent_id)
    uhd_id = str(uuid.uuid4())
    model_dir = _make_model_dir(tmp_path / "model", uhd_id)

    with caplog.at_level("WARNING"):
        assert (
            main(["promote", "--model-dir", str(model_dir), "--descriptor-tree", str(tree)]) == 0
        )

    assert "OVERWRITING" not in caplog.text
    assert _heuristic_of(ued) == uhd_id


def test_promote_is_idempotent_when_the_pair_is_already_in_place(tmp_path):
    """Retraining straight into the tree: nothing to copy, but the id still has to be
    written, and copying a file onto itself must not raise."""
    tree = tmp_path / "tree"
    uhd_id = str(uuid.uuid4())
    _make_model_dir(tree, uhd_id)
    ued = _make_ued(tree / "a.ued.json", "hipkernel:pointwise", UED_ID_A)

    assert main(["promote", "--model-dir", str(tree), "--descriptor-tree", str(tree)]) == 0

    assert _heuristic_of(ued) == uhd_id
    assert (tree / "model.bin").is_file()


# --------------------------------------------------------------------------------
# The loop the two pieces close
# --------------------------------------------------------------------------------


def test_retraining_with_the_promoted_id_needs_no_second_promote(tmp_path):
    """train -> promote -> train --uhd-id <same>: the UED never has to change again."""
    csv = _training_csv(tmp_path / "bench.csv")
    output_dir = tmp_path / "model"
    tree = tmp_path / "tree"
    ued = _make_ued(tree / "engine.ued.json", "hipkernel:pointwise", UED_ID_A)

    assert _train(output_dir, csv) == 0
    assert main(["promote", "--model-dir", str(output_dir), "--descriptor-tree", str(tree)]) == 0
    promoted_id = _heuristic_of(ued)
    after_promote = ued.read_text(encoding="utf-8")

    assert _train(output_dir, csv, "--uhd-id", promoted_id) == 0
    assert main(["promote", "--model-dir", str(output_dir), "--descriptor-tree", str(tree)]) == 0

    assert ued.read_text(encoding="utf-8") == after_promote
    assert _heuristic_of(ued) == promoted_id
    installed = json.loads((tree / "heuristic.uhd.json").read_text(encoding="utf-8"))
    assert installed["id"] == promoted_id
