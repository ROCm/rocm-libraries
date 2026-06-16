# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

"""Tests for ``metrics._artifact_paths.flatten_hostname_dir``.

The helper exists because rocprofv3 always nests its output under a
``<hostname>/`` segment with no flag to suppress it. For single-host
runs that segment is pure noise in the artifact path the user sees.
"""

from pathlib import Path

from dnn_benchmarking.metrics._artifact_paths import (
    _strip_results_infix,
    flatten_hostname_dir,
)


class TestStripResultsInfix:
    def test_strips_results_infix(self):
        # rocprofv3 produces `<-o value>_results.<ext>`; with `-o results`
        # that becomes `results_results.db` — strip back to `results.db`.
        assert _strip_results_infix("results_results.db") == "results.db"
        assert _strip_results_infix("results_results.pftrace") == "results.pftrace"

    def test_leaves_unrelated_names_alone(self):
        assert _strip_results_infix("results.db") == "results.db"
        assert _strip_results_infix("counters.csv") == "counters.csv"
        # Has `_results.` but stem doesn't end with `_results` — not our pattern.
        assert _strip_results_infix("foo_results.bar.txt") == "foo_results.bar.txt"

    def test_no_extension_returns_unchanged(self):
        # Edge: filename without an extension is rocprofv3-uncharacteristic
        # but the helper shouldn't crash on it.
        assert _strip_results_infix("readme") == "readme"


class TestFlattenHostnameDir:
    def test_hoists_files_out_of_single_hostname_subdir(self, tmp_path):
        """The mainline case: rocprofv3 wrote one ``<hostname>/`` dir
        containing the artifact files. After flattening, files sit
        directly under ``out_dir`` and the hostname dir is gone."""
        host = tmp_path / "host-1"
        host.mkdir()
        # rocprofv3-shaped filenames with the _results infix.
        (host / "results_results.db").write_bytes(b"sqlite")
        (host / "results_results.pftrace").write_bytes(b"trace")

        flatten_hostname_dir(tmp_path)

        # Hoisted AND suffix-stripped.
        assert (tmp_path / "results.db").read_bytes() == b"sqlite"
        assert (tmp_path / "results.pftrace").read_bytes() == b"trace"
        assert not (tmp_path / "results_results.db").exists()
        assert not host.exists()

    def test_noop_when_out_dir_missing(self, tmp_path):
        """Defensive — sources may bail before rocprofv3 writes
        anything (binary missing, OSError on spawn). Calling on a
        non-existent dir must not raise."""
        flatten_hostname_dir(tmp_path / "does-not-exist")
        # No assertion needed — must not raise.

    def test_strips_results_infix_from_top_level_files(self, tmp_path):
        """When rocprofv3 is given an explicit ``-o``, it writes
        directly to ``<out_dir>/<basename>_results.<ext>`` with no
        hostname segment. The helper must still strip the suffix in
        this case — otherwise the user-facing path keeps the
        ``_results`` infix."""
        (tmp_path / "results_results.db").write_bytes(b"sqlite")
        (tmp_path / "results_results.pftrace").write_bytes(b"trace")
        flatten_hostname_dir(tmp_path)
        assert (tmp_path / "results.db").read_bytes() == b"sqlite"
        assert (tmp_path / "results.pftrace").read_bytes() == b"trace"
        assert not (tmp_path / "results_results.db").exists()

    def test_leaves_unrelated_top_level_files_alone(self, tmp_path):
        """Files at top level without the ``_results`` pattern stay put
        with their original name."""
        (tmp_path / "loose.txt").write_text("don't move me")
        flatten_hostname_dir(tmp_path)
        assert (tmp_path / "loose.txt").read_text() == "don't move me"

    def test_leaves_nonempty_nested_dirs_alone(self, tmp_path):
        """If a hostname subdir contains further nested directories
        (unexpected — rocprofv3 doesn't do this on supported versions,
        but a future rocprofv3 might), leave the hostname dir alone so
        nothing is silently dropped. Top-level files in the hostname
        dir still get hoisted."""
        host = tmp_path / "somehost"
        host.mkdir()
        (host / "results.db").write_bytes(b"db")
        (host / "extra").mkdir()
        (host / "extra" / "weird.txt").write_text("unexpected")

        flatten_hostname_dir(tmp_path)

        # File hoisted, but the non-empty hostname dir stays.
        assert (tmp_path / "results.db").read_bytes() == b"db"
        assert host.exists()
        assert (host / "extra" / "weird.txt").read_text() == "unexpected"

    def test_handles_multiple_hostname_dirs(self, tmp_path):
        """Edge case: if a rerun left a stale second hostname dir
        alongside the current one (e.g. cluster reschedule), both
        get flattened. Last-writer-wins on filename collisions."""
        host1 = tmp_path / "h1"
        host2 = tmp_path / "h2"
        host1.mkdir()
        host2.mkdir()
        (host1 / "a.db").write_bytes(b"a-from-h1")
        (host2 / "b.db").write_bytes(b"b-from-h2")

        flatten_hostname_dir(tmp_path)

        assert (tmp_path / "a.db").read_bytes() == b"a-from-h1"
        assert (tmp_path / "b.db").read_bytes() == b"b-from-h2"
        assert not host1.exists()
        assert not host2.exists()
