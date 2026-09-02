# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""The embedded source check, run the way the build runs it.

The tool takes emitted JSON and a key table, so every case here writes both by
hand and invokes the script as a subprocess. Nothing imports the packer: a test
that recomputed a key from the packer would pass on two sides of one mistake.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

TOOL = Path(__file__).resolve().parents[1] / "tools" / "hkp_verify_embedded_sources.py"

TARGET = "hip_kernel_provider_tests"
ARCH = "gfx942"
OTHER_ARCH = "gfx90a"
KEY = "kernels/PointwiseAdd.cpp"
LABEL = "unit_pointwise"


def _kernel_source(key):
    return {
        "kind": "embedded_source",
        "source_file": key,
        "entry_point": "PointwiseAdd",
    }


def _provenance(rel_dir, authored, label=LABEL):
    provenance = {"origin_kind": "embedded_source"}
    if label is not None:
        provenance["source_label"] = label
    provenance.update(
        {
            "rel_dir": rel_dir,
            "source_file": authored,
            "authored_arch": [],
            "rewritten": ["arch"],
        }
    )
    return provenance


def _write(path, doc):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    return path


def _ukd(shard, name, key, rel_dir=".", authored=None, provenance=True, label=LABEL):
    """A standalone UKD, which carries both blocks at its document root."""
    doc = {
        "version": "1.0",
        "id": name,
        "name": name,
        "kernel_source": _kernel_source(key),
        "metadata": {},
        "priority": 0,
    }
    if provenance:
        doc["provenance"] = _provenance(
            rel_dir, key if authored is None else authored, label=label
        )
    doc["arch"] = [shard.name]
    return _write(shard / f"{name}.ukd.json", doc)


def _kdp(shard, name, keys, rel_dir="."):
    """A KDP, whose provenance sits on each inline entry and not at its root."""
    entries = [
        {
            "version": "1.0",
            "id": f"{name}-{index}",
            "name": f"{name}.{index}",
            "kernel_source": _kernel_source(key),
            "metadata": {"block_size": 64},
            "priority": 0,
            "provenance": _provenance(rel_dir, key),
            "arch": [shard.name],
        }
        for index, key in enumerate(keys)
    ]
    doc = {
        "version": "1.0",
        "id": name,
        "name": name,
        "arch": [shard.name],
        "matchers": [],
        "engine": "engine-id",
        "dispatch": "dispatch-id",
        "kernelDescriptors": entries + ["a-referenced-standalone-id"],
    }
    return _write(shard / f"{name}.kdp.json", doc)


def _manifest(tmp_path, pairs):
    """Write a key table the way embed_kernel_sources() writes it."""
    path = tmp_path / f"{TARGET}_kernel_keys.txt"
    path.write_text(
        "".join(f"{key}\t{registered}\n" for key, registered in sorted(pairs)),
        encoding="utf-8",
    )
    return path


def _source_root(tmp_path, name="authored"):
    """The absolute source root one pack label maps to."""
    return str(tmp_path / name)


def _source(tmp_path, *parts, root="authored"):
    """The absolute path a key registers, under the root its label maps to."""
    return str(tmp_path.joinpath(root, *parts))


def _authored(tmp_path, *parts, root="authored"):
    """The authored location the check spells, whose separators it unifies."""
    return _source(tmp_path, *parts, root=root).replace("\\", "/")


def _labels(tmp_path, **extra):
    """The label -> source root map the build passes for every wired pack."""
    return {LABEL: _source_root(tmp_path), **extra}


def _drop_anchor(path):
    """Respell one absolute path relative, keeping every segment it names."""
    return Path(path).relative_to(Path(path).anchor).as_posix()


def _run(manifest, roots, source_roots, target=TARGET):
    argv = [
        sys.executable,
        str(TOOL),
        "--target",
        target,
        "--key-manifest",
        str(manifest),
    ]
    for root in roots:
        argv += ["--staged-descriptor-root", str(root)]
    for label, source_root in sorted(source_roots.items()):
        argv += ["--source-root", f"{label}={source_root}"]
    return subprocess.run(argv, capture_output=True, text=True)


def _count_line(checked, keys, target=TARGET):
    """The one line a pass reports, which is the evidence the step ran."""
    return (
        f"hkp_verify_embedded_sources: {target}: {checked} embedded_source "
        f"descriptors checked against {keys} table keys"
    )


@pytest.mark.quick
def test_a_table_and_a_tree_that_agree_pass(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _count_line(1, 1)


@pytest.mark.quick
def test_a_pass_reports_what_it_compared(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    other_key = "kernels/PointwiseMul.cpp"
    _kdp(root / ARCH, "pointwise", [KEY, other_key])
    _ukd(root / ARCH, "pointwise_add_f16", KEY)
    manifest = _manifest(
        tmp_path,
        [
            (KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp")),
            (other_key, _source(tmp_path, "kernels", "PointwiseMul.cpp")),
        ],
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 0, result.stderr
    # Three descriptors against two keys: the counts are independent, and the
    # bare id in the KDP's kernelDescriptors is not one of them.
    assert result.stdout.strip() == _count_line(3, 2)


@pytest.mark.quick
def test_a_vacuous_pass_reports_both_counts_as_zero(tmp_path):
    target = "hip_kernel_provider_integration_tests"
    root = tmp_path / "integration" / "conv"
    root.mkdir(parents=True)

    result = _run(tmp_path / "no_such_kernel_keys.txt", [root], {}, target=target)

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _count_line(0, 0, target=target)


@pytest.mark.quick
def test_a_key_the_table_lacks_fails(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    descriptor = _ukd(root / ARCH, "pointwise_add", KEY)
    manifest = _manifest(tmp_path, [])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert KEY in result.stderr
    assert descriptor.name in result.stderr
    assert TARGET in result.stderr
    assert "clean build directory" in result.stderr


@pytest.mark.quick
def test_a_table_entry_no_descriptor_names_passes(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    manifest = _manifest(
        tmp_path,
        [
            (KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp")),
            ("vector_add.cpp", _source(tmp_path, "vector_add.cpp")),
        ],
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 0, result.stderr


@pytest.mark.quick
def test_an_empty_table_and_an_empty_root_pass(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    root.mkdir(parents=True)
    manifest = _manifest(tmp_path, [])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 0, result.stderr


@pytest.mark.quick
def test_a_root_that_is_not_there_passes(tmp_path):
    manifest = _manifest(tmp_path, [])

    result = _run(manifest, [tmp_path / "never" / "staged"], _labels(tmp_path))

    assert result.returncode == 0, result.stderr


@pytest.mark.quick
def test_a_table_that_is_not_there_passes(tmp_path):
    root = tmp_path / "integration" / "conv"
    root.mkdir(parents=True)

    result = _run(tmp_path / "no_such_kernel_keys.txt", [root], _labels(tmp_path))

    assert result.returncode == 0, result.stderr


@pytest.mark.quick
def test_the_check_spans_every_arch_shard(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    other_key = "kernels/PointwiseMul.cpp"
    _ukd(root / ARCH, "pointwise_add", KEY)
    _ukd(root / OTHER_ARCH, "pointwise_mul", other_key)
    pairs = [
        (KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp")),
        (other_key, _source(tmp_path, "kernels", "PointwiseMul.cpp")),
    ]

    assert _run(_manifest(tmp_path, pairs), [root], _labels(tmp_path)).returncode == 0

    result = _run(_manifest(tmp_path, pairs[:1]), [root], _labels(tmp_path))

    assert result.returncode == 1
    assert other_key in result.stderr


@pytest.mark.quick
def test_a_staging_directory_is_not_read(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    _ukd(root / f".{ARCH}.staging", "retracted", "kernels/Retracted.cpp")
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 0, result.stderr
    # One descriptor, not two: the count names what the walk reached.
    assert result.stdout.strip() == _count_line(1, 1)


@pytest.mark.quick
def test_a_staging_directory_below_the_root_is_not_read(tmp_path):
    """The skip reads every parent segment, so it holds at any depth."""
    parent = tmp_path / "unit"
    pack = parent / "pointwise"
    _ukd(pack / ARCH, "pointwise_add", KEY)
    _ukd(pack / f".{ARCH}.staging", "retracted", "kernels/Retracted.cpp")
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [parent], _labels(tmp_path))

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _count_line(1, 1)


@pytest.mark.quick
def test_a_source_registered_from_another_folder_fails(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    key = "a/kernels/Foo.cpp"
    descriptor = _ukd(root / ARCH, "foo", key, rel_dir="a", authored="kernels/Foo.cpp")
    registered = _source(tmp_path, "b", "kernels", "Foo.cpp")
    manifest = _manifest(tmp_path, [(key, registered)])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert "a/kernels/Foo.cpp" in result.stderr
    assert registered in result.stderr
    assert _authored(tmp_path, "a", "kernels", "Foo.cpp") in result.stderr
    assert descriptor.name in result.stderr
    assert "clean build directory" in result.stderr


@pytest.mark.quick
def test_a_folder_whose_name_merely_starts_the_same_fails(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    key = "a/kernels/Foo.cpp"
    _ukd(root / ARCH, "foo", key, rel_dir="a", authored="kernels/Foo.cpp")
    manifest = _manifest(
        tmp_path, [(key, _source(tmp_path, "xa", "kernels", "Foo.cpp"))]
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert "a/kernels/Foo.cpp" in result.stderr


@pytest.mark.quick
def test_a_root_relative_descriptor_is_still_placed_exactly(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY, rel_dir=".")
    registered = _source(tmp_path, "any", "where", "kernels", "PointwiseAdd.cpp")
    manifest = _manifest(tmp_path, [(KEY, registered)])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert registered in result.stderr
    assert _authored(tmp_path, "kernels", "PointwiseAdd.cpp") in result.stderr


@pytest.mark.quick
def test_a_key_registered_from_the_build_tree_fails(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    registered = _source(tmp_path, "kernels", "PointwiseAdd.cpp", root="build")
    manifest = _manifest(tmp_path, [(KEY, registered)])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert registered in result.stderr
    assert _authored(tmp_path, "kernels", "PointwiseAdd.cpp") in result.stderr


@pytest.mark.quick
def test_a_case_only_difference_fails(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    registered = _source(tmp_path, "Kernels", "PointwiseAdd.cpp")
    manifest = _manifest(tmp_path, [(KEY, registered)])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert registered in result.stderr


@pytest.mark.quick
def test_two_labels_with_two_source_roots_resolve_apart(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    key = "kernels/Foo.cpp"
    _ukd(root / ARCH, "foo_a", key)
    other = _ukd(root / ARCH, "foo_b", key, label="unit_conv")
    registered = _source(tmp_path, "kernels", "Foo.cpp")
    manifest = _manifest(tmp_path, [(key, registered)])

    result = _run(
        manifest,
        [root],
        _labels(tmp_path, unit_conv=_source_root(tmp_path, "other")),
    )

    assert result.returncode == 1
    assert other.name in result.stderr
    assert registered in result.stderr
    assert _authored(tmp_path, "kernels", "Foo.cpp", root="other") in result.stderr


@pytest.mark.quick
def test_two_labels_may_share_one_source_root(tmp_path):
    root = tmp_path / "unit" / "conv"
    key = "kernels/Conv.cpp"
    other_key = "kernels/ConvBwd.cpp"
    _ukd(root / ARCH, "conv", key, label="unit_conv")
    _ukd(root / ARCH, "conv_bwd", other_key, label="integration_conv")
    manifest = _manifest(
        tmp_path,
        [
            (key, _source(tmp_path, "kernels", "Conv.cpp")),
            (other_key, _source(tmp_path, "kernels", "ConvBwd.cpp")),
        ],
    )

    result = _run(
        manifest,
        [root],
        {
            "unit_conv": _source_root(tmp_path),
            "integration_conv": _source_root(tmp_path),
        },
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == _count_line(2, 2)


@pytest.mark.quick
def test_a_registered_path_with_a_dot_dot_segment_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    registered = _source(tmp_path, "kernels", "..", "kernels", "PointwiseAdd.cpp")
    manifest = _manifest(tmp_path, [(KEY, registered)])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert "'..'" in result.stderr
    assert registered in result.stderr


@pytest.mark.quick
def test_a_source_root_with_a_dot_dot_segment_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )
    source_root = str(tmp_path / "elsewhere" / ".." / "authored")

    result = _run(manifest, [root], {LABEL: source_root})

    assert result.returncode == 1
    assert "'..'" in result.stderr


@pytest.mark.quick
def test_a_registered_path_that_is_not_absolute_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    # Every segment of the absolute path, without the anchor that makes it one.
    registered = _drop_anchor(_source(tmp_path, "kernels", "PointwiseAdd.cpp"))
    manifest = _manifest(tmp_path, [(KEY, registered)])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert "is not an absolute path" in result.stderr
    assert registered in result.stderr


@pytest.mark.quick
def test_a_source_root_that_is_not_absolute_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [root], {LABEL: _drop_anchor(_source_root(tmp_path))})

    assert result.returncode == 1
    assert "is not an absolute path" in result.stderr


@pytest.mark.quick
def test_a_descriptor_without_a_source_label_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    descriptor = _ukd(root / ARCH, "pointwise_add", KEY, label=None)
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert descriptor.name in result.stderr
    assert "provenance.source_label" in result.stderr
    assert "provenance.rel_dir" not in result.stderr
    assert "clean build directory" in result.stderr


@pytest.mark.quick
def test_a_source_label_the_map_does_not_know_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    descriptor = _ukd(root / ARCH, "pointwise_add", KEY, label="unit_extra")
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert "unit_extra" in result.stderr
    assert descriptor.name in result.stderr
    # The labels it does know, which is what makes the wiring fault one read.
    assert LABEL in result.stderr
    # A wiring fault, not a stale tree: the hint would send the reader elsewhere.
    assert "clean build directory" not in result.stderr


@pytest.mark.quick
def test_a_malformed_source_root_pair_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    _ukd(root / ARCH, "pointwise_add", KEY)
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    argv = [
        sys.executable,
        str(TOOL),
        "--target",
        TARGET,
        "--key-manifest",
        str(manifest),
        "--staged-descriptor-root",
        str(root),
        "--source-root",
        _source_root(tmp_path),
    ]
    result = subprocess.run(argv, capture_output=True, text=True)

    assert result.returncode == 1
    assert "'<label>=<path>'" in result.stderr


@pytest.mark.quick
def test_a_descriptor_without_provenance_is_an_error(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    descriptor = _ukd(root / ARCH, "pointwise_add", KEY, provenance=False)
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert descriptor.name in result.stderr
    assert "provenance.rel_dir" in result.stderr
    assert "provenance.source_file" in result.stderr
    assert "provenance.source_label" in result.stderr


@pytest.mark.quick
def test_a_descriptor_without_provenance_outranks_a_key_the_table_lacks(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    descriptor = _ukd(root / ARCH, "pointwise_add", KEY, provenance=False)
    manifest = _manifest(tmp_path, [])

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert descriptor.name in result.stderr
    assert "provenance.rel_dir" in result.stderr
    # A descriptor that records no authored location cannot have one checked.
    assert "embeds no source under the key" not in result.stderr


@pytest.mark.quick
def test_each_inline_entry_of_a_kdp_is_read_with_its_own_provenance(tmp_path):
    root = tmp_path / "unit" / "pointwise"
    other_key = "kernels/PointwiseMul.cpp"
    descriptor = _kdp(root / ARCH, "pointwise", [KEY, other_key])
    manifest = _manifest(
        tmp_path, [(KEY, _source(tmp_path, "kernels", "PointwiseAdd.cpp"))]
    )

    result = _run(manifest, [root], _labels(tmp_path))

    assert result.returncode == 1
    assert other_key in result.stderr
    assert descriptor.name in result.stderr
    # The presence miss, not the missing-provenance error: reading the KDP root for
    # provenance would report the latter for every entry, including the matching one.
    assert "embeds no source under the key" in result.stderr
    assert "provenance" not in result.stderr
