# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/prior_ids.py: a corrupted prior bundle is refused with a
message naming the problem, and the DVC pointer check hashes tracked files the way
DVC records them."""

import hashlib
import json
import uuid

import pytest

from codegen.generator import resolve_ids
from codegen.prior_ids import dvc_mismatches, unfetched_dvc
from tests.helpers import home_and_satellite


def _edit(path, change):
    doc = json.loads(path.read_text(encoding="utf-8"))
    change(doc)
    path.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")


def _id_of(path):
    return json.loads(path.read_text(encoding="utf-8"))["id"]


def _copy_kdp_with_new_ids(directory, file="other.kdp.json", **changes):
    doc = json.loads((directory / "test.kdp.json").read_text(encoding="utf-8"))
    doc["id"] = str(uuid.uuid4())
    for entry in doc["kernelDescriptors"]:
        entry["id"] = str(uuid.uuid4())
    doc.update(changes)
    (directory / file).write_text(json.dumps(doc), encoding="utf-8")


def _non_string_id(directory):
    (directory / "stray.umd.json").write_text(json.dumps({"id": 5, "scope": "graph"}))


def _repeated_id(directory):
    ued_id = _id_of(directory / "test.ued.json")
    _edit(
        directory / "test.kdp.json",
        lambda kdp: kdp["kernelDescriptors"][0].__setitem__("id", ued_id),
    )


def _non_object_document(directory):
    (directory / "stray.umd.json").write_text("[]")


def _non_object_kernel(directory):
    _edit(
        directory / "test.kdp.json",
        lambda kdp: kdp["kernelDescriptors"].append(5),
    )


def _dispatch_of_wrong_kind(directory):
    kmd_id = _id_of(directory / "test.kmd.json")
    _edit(directory / "test.kdp.json", lambda kdp: kdp.__setitem__("dispatch", kmd_id))


def _two_kernel_scope_matchers(directory):
    _edit(
        directory / "test.kdp.json",
        lambda kdp: kdp.__setitem__("matchers", kdp["matchers"] * 2),
    )


def _no_kernel_scope_matcher(directory):
    _edit(directory / "test.kdp.json", lambda kdp: kdp.__setitem__("matchers", []))


def _repeated_kdp_name(directory):
    _copy_kdp_with_new_ids(directory)


def _unparseable_document(directory):
    (directory / "stray.umd.json").write_text("{", encoding="utf-8")


def _no_dispatch(directory):
    _edit(directory / "test.kdp.json", lambda kdp: kdp.pop("dispatch"))


def _catalog_without_default(directory):
    _edit(
        directory / "test.ued.json",
        lambda ued: ued.__setitem__("sort_kernel_catalog", {}),
    )


def _kdp_arch_not_a_list(directory):
    _edit(directory / "test.kdp.json", lambda kdp: kdp.__setitem__("arch", "gfx950"))


def _second_kdp(matchers):
    """A second KDP of the engine whose matchers are ``matchers(directory, kernel
    matcher id)``. It sorts after ``test.kdp.json``, so the engine-level documents
    still resolve from the first and the check reaches the second."""

    def corrupt(directory):
        first = json.loads((directory / "test.kdp.json").read_text(encoding="utf-8"))
        (kernel_matcher,) = first["matchers"]
        _copy_kdp_with_new_ids(
            directory,
            "zz.kdp.json",
            name="zz",
            matchers=matchers(directory, kernel_matcher),
        )

    return corrupt


def _kernel_scope_umd(directory):
    identity = str(uuid.uuid4())
    (directory / "stray.umd.json").write_text(
        json.dumps({"id": identity, "scope": "kernel"}), encoding="utf-8"
    )
    return identity


def _kernel(change):
    """Applies ``change`` to the first kernel entry of the KDP."""

    def corrupt(directory):
        _edit(
            directory / "test.kdp.json",
            lambda kdp: change(kdp["kernelDescriptors"][0]),
        )

    return corrupt


def _as_sources(sources):
    """Replaces the kernel's ``kernel_source`` with ``sources(kernel_source)``."""

    def change(kernel):
        kernel["sources"] = sources(kernel.pop("kernel_source"))

    return change


_NO_KERNEL_SOURCE = "has no 'kernel_source' object"
_BAD_SOURCES = "needs 'sources' to be a non-empty object"


@pytest.mark.parametrize(
    ("corrupt", "message"),
    [
        pytest.param(_non_string_id, "has no string 'id'", id="non-string-id"),
        pytest.param(_repeated_id, "appears twice under", id="repeated-id"),
        pytest.param(
            _unparseable_document, "cannot read descriptor", id="unparseable-document"
        ),
        pytest.param(
            _non_object_document, "must be an object", id="non-object-document"
        ),
        pytest.param(
            _non_object_kernel, "kernel entry is not an object", id="non-object-kernel"
        ),
        pytest.param(
            _dispatch_of_wrong_kind, "is not a udd descriptor", id="wrong-kind-dispatch"
        ),
        pytest.param(_no_dispatch, "disagree on dispatch", id="no-dispatch"),
        pytest.param(
            _catalog_without_default,
            "must be an object with a 'default'",
            id="catalog-without-default",
        ),
        pytest.param(
            _two_kernel_scope_matchers,
            "two kernel-scope matchers",
            id="two-same-scope-matchers",
        ),
        pytest.param(
            _no_kernel_scope_matcher,
            "no kernel-scope matcher among",
            id="no-kernel-scope-matcher",
        ),
        pytest.param(
            _second_kdp(lambda directory, kernel: []),
            "do not include the engine's kernel-scope matcher",
            id="second-kdp-without-the-kernel-matcher",
        ),
        pytest.param(
            _second_kdp(
                lambda directory, kernel: [
                    kernel,
                    str(uuid.uuid4()),
                    str(uuid.uuid4()),
                ]
            ),
            "2 matchers besides the kernel-scope one",
            id="second-kdp-with-two-operation-matchers",
        ),
        pytest.param(
            _second_kdp(
                lambda directory, kernel: [kernel, _kernel_scope_umd(directory)]
            ),
            "is not graph-scope",
            id="second-kdp-operation-matcher-not-graph-scope",
        ),
        pytest.param(_repeated_kdp_name, "two KDPs named", id="repeated-kdp-name"),
        pytest.param(
            _kdp_arch_not_a_list, "'arch' is not a list of strings", id="kdp-arch"
        ),
        pytest.param(
            _kernel(lambda kernel: kernel.__setitem__("arch", "gfx950")),
            "'arch' that is not a list of strings",
            id="kernel-arch",
        ),
        pytest.param(
            _kernel(lambda kernel: kernel.__setitem__("metadata", [1])),
            "non-object 'metadata'",
            id="non-object-metadata",
        ),
        pytest.param(
            _kernel(lambda kernel: kernel.__setitem__("kernel_source", 5)),
            _NO_KERNEL_SOURCE,
            id="non-object-kernel-source",
        ),
        pytest.param(
            _kernel(lambda kernel: kernel.pop("kernel_source")),
            _NO_KERNEL_SOURCE,
            id="no-source",
        ),
        pytest.param(
            _kernel(_as_sources(lambda source: {})), _BAD_SOURCES, id="empty-sources"
        ),
        pytest.param(
            _kernel(_as_sources(lambda source: {"a": 5})),
            _BAD_SOURCES,
            id="non-object-launch",
        ),
        pytest.param(
            _kernel(
                lambda kernel: kernel.__setitem__(
                    "sources", {"a": kernel["kernel_source"]}
                )
            ),
            _BAD_SOURCES,
            id="both-source-keys",
        ),
        pytest.param(
            _kernel(lambda kernel: kernel["kernel_source"].__setitem__("spec", [1])),
            "'spec' that is not an object",
            id="non-object-spec",
        ),
        pytest.param(
            _kernel(_as_sources(lambda source: {"a": {**source, "spec": [1]}})),
            "'spec' that is not an object",
            id="non-object-launch-spec",
        ),
    ],
)
def test_corrupted_prior_bundle_is_refused(generator, tmp_path, corrupt, message):
    bundles = home_and_satellite(generator, tmp_path)
    corrupt(bundles.home_dir)
    with pytest.raises(ValueError, match=message):
        resolve_ids(bundles.home, bundles.home_root, ids_from=bundles.home_dir)


def test_a_previous_bundle_that_is_not_a_directory_is_refused(generator, tmp_path):
    bundles = home_and_satellite(generator, tmp_path)
    gone = tmp_path / "gone" / bundles.home.descriptor_dir
    with pytest.raises(ValueError, match="is not a directory"):
        resolve_ids(bundles.home, tmp_path / "out", ids_from=gone)


def _write_pointer(directory, name, target, md5):
    (directory / f"{name}.dvc").write_text(
        f"outs:\n- md5: {md5}\n  path: {target}\n", encoding="utf-8"
    )


class TestDvcMismatches:
    def test_crlf_checkout_of_a_tracked_text_file_is_not_stale(self, tmp_path):
        text = b'{"a": 1}\n{"b": 2}\n'
        (tmp_path / "x.kdp.json").write_bytes(text.replace(b"\n", b"\r\n"))
        _write_pointer(
            tmp_path, "x.kdp.json", "x.kdp.json", hashlib.md5(text).hexdigest()
        )
        assert dvc_mismatches(tmp_path) == []

    def test_edited_tracked_file_is_stale(self, tmp_path):
        text = b"one\n"
        (tmp_path / "x.txt").write_bytes(b"two\n")
        _write_pointer(tmp_path, "x.txt", "x.txt", hashlib.md5(text).hexdigest())
        [message] = dvc_mismatches(tmp_path)
        assert "x.txt" in message and "stale or edited" in message

    def test_binary_file_is_hashed_byte_for_byte(self, tmp_path):
        data = b"\0\r\n\0"
        (tmp_path / "x.bin").write_bytes(data)
        _write_pointer(tmp_path, "x.bin", "x.bin", hashlib.md5(data).hexdigest())
        assert dvc_mismatches(tmp_path) == []

    @staticmethod
    def _tracked_directory(tmp_path):
        tree = tmp_path / "tree"
        (tree / "a").mkdir(parents=True)
        (tree / "a.txt").write_bytes(b"alpha\n")
        (tree / "a" / "b.txt").write_bytes(b"beta\n")
        manifest = [
            {"md5": hashlib.md5(b"alpha\n").hexdigest(), "relpath": "a.txt"},
            {"md5": hashlib.md5(b"beta\n").hexdigest(), "relpath": "a/b.txt"},
        ]
        recorded = hashlib.md5(json.dumps(manifest).encode()).hexdigest() + ".dir"
        _write_pointer(tmp_path, "tree", "tree", recorded)
        return tree

    def test_matching_directory_is_not_stale(self, tmp_path):
        self._tracked_directory(tmp_path)
        assert dvc_mismatches(tmp_path) == []

    def test_modified_file_in_a_tracked_directory_is_stale(self, tmp_path):
        tree = self._tracked_directory(tmp_path)
        (tree / "a" / "b.txt").write_bytes(b"changed\n")
        [message] = dvc_mismatches(tmp_path)
        assert "tree" in message and "stale or edited" in message

    @pytest.mark.parametrize(
        "pointer",
        ["outs: [", "- a\n", "outs: 5\n", "outs:\n- 5\n- path: 7\n"],
        ids=["unparseable", "not-a-mapping", "outs-not-a-list", "out-without-a-path"],
    )
    def test_a_malformed_pointer_tracks_nothing(self, tmp_path, pointer):
        """A pointer DVC did not write is skipped by both readers, not raised on;
        a well-formed pointer beside it is still read."""
        bundle = tmp_path / "c"
        bundle.mkdir()
        (bundle / "bad.dvc").write_text(pointer, encoding="utf-8")
        _write_pointer(bundle, "good", "missing.kdp.json", "0" * 32)

        assert dvc_mismatches(bundle) == []
        [message] = unfetched_dvc(tmp_path, [])
        assert "good.dvc" in message and "missing.kdp.json" in message


class TestEngineHeuristicReference:
    """The UED references its UHD as ``sort_kernel_catalog.default``."""

    def test_a_regeneration_keeps_the_heuristic_descriptors_id(
        self, generator, tmp_path
    ):
        bundles = home_and_satellite(generator, tmp_path)
        (ued_path,) = bundles.home_dir.glob("*.ued.json")
        (uhd_path,) = bundles.home_dir.glob("*.uhd.json")
        ued = json.loads(ued_path.read_text(encoding="utf-8"))
        assert ued["sort_kernel_catalog"] == {"default": _id_of(uhd_path)}

        resolution = resolve_ids(
            bundles.home, tmp_path / "out", ids_from=bundles.home_dir
        )

        assert resolution.ids["uhd"] == _id_of(uhd_path)
        assert resolution.report.minted == 0

    def test_a_default_that_is_not_a_uhd_is_refused(self, generator, tmp_path):
        bundles = home_and_satellite(generator, tmp_path)
        (ued_path,) = bundles.home_dir.glob("*.ued.json")
        _edit(
            ued_path,
            lambda ued: ued.__setitem__(
                "sort_kernel_catalog", {"default": ued["metadata"]}
            ),
        )

        with pytest.raises(ValueError, match="sort_kernel_catalog.default"):
            resolve_ids(bundles.home, tmp_path / "out", ids_from=bundles.home_dir)
