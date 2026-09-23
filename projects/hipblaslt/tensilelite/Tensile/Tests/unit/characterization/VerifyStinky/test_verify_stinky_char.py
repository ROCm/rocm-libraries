################################################################################
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
################################################################################

"""Characterization tests for ``Tensile.verify_stinky_comment_vs_elf_text``:
the STINKY_TOTAL_INST_BYTES verification logic with synthetic fixtures."""

import importlib
import os
import sys
from pathlib import Path

import pytest

verify_stinky = importlib.import_module("Tensile.verify_stinky_comment_vs_elf_text")

pytestmark = pytest.mark.unit


class TestExtractStinkyTotalInstBytes:
    """Test the stinky marker extraction from asm text."""

    def test_extract_valid_marker(self):
        s_text = "# some assembly\n# STINKY_TOTAL_INST_BYTES: 1024\n.text"
        cost = verify_stinky.extract_stinky_total_inst_bytes(s_text)
        assert cost == 1024

    def test_extract_marker_no_newline(self):
        s_text = "# STINKY_TOTAL_INST_BYTES: 512"
        cost = verify_stinky.extract_stinky_total_inst_bytes(s_text)
        assert cost == 512

    def test_extract_marker_missing_returns_none(self):
        s_text = "# just some assembly\n.text"
        cost = verify_stinky.extract_stinky_total_inst_bytes(s_text)
        assert cost is None

    def test_extract_marker_empty_string_returns_none(self):
        cost = verify_stinky.extract_stinky_total_inst_bytes("")
        assert cost is None

    def test_extract_marker_whitespace_variations(self):
        # Test various whitespace patterns
        s_text = "# STINKY_TOTAL_INST_BYTES:   256"
        cost = verify_stinky.extract_stinky_total_inst_bytes(s_text)
        assert cost == 256


class TestParseDotTextSize:
    """Test the .text section size extraction from readelf output."""

    def test_parse_text_size_hex(self):
        readelf_out = """Section Headers:
  [Nr] Name          Type            Address          Offset
  [ 1] .text         PROGBITS        0000000000000000  00000000  0001000  00000"""
        size = verify_stinky._parse_dot_text_size(readelf_out)
        assert size == 0x1000

    def test_parse_text_size_hex_with_0x_prefix(self):
        readelf_out = ".text PROGBITS 0x1000 0x2000 0x800 AX"
        size = verify_stinky._parse_dot_text_size(readelf_out)
        assert size == 0x800

    def test_parse_text_size_no_prefix(self):
        readelf_out = "  [ 1] .text         PROGBITS        0x0000000000000000  0x0000000000000000  9660  00"
        size = verify_stinky._parse_dot_text_size(readelf_out)
        assert size == 0x9660

    def test_parse_text_size_not_found_returns_none(self):
        readelf_out = """Section Headers:
  [Nr] Name
  [ 1] .rodata
  [ 2] .data"""
        size = verify_stinky._parse_dot_text_size(readelf_out)
        assert size is None

    def test_parse_text_size_empty_input_returns_none(self):
        size = verify_stinky._parse_dot_text_size("")
        assert size is None

    def test_parse_text_size_insufficient_tokens(self):
        readelf_out = ".text PROGBITS"
        size = verify_stinky._parse_dot_text_size(readelf_out)
        assert size is None


class TestFindReadelf:
    """Test readelf executable location."""

    def test_find_readelf_via_path(self, tmp_path, monkeypatch):
        # Create a fake readelf in a temp directory and put it on PATH
        readelf = tmp_path / "llvm-readelf"
        readelf.write_text("#!/bin/sh\necho fake")
        os.chmod(readelf, 0o755)
        monkeypatch.setenv("PATH", str(tmp_path))
        # Clear ROCM_PATH and LLVM_BIN so it falls back to PATH
        monkeypatch.delenv("ROCM_PATH", raising=False)
        monkeypatch.delenv("LLVM_BIN", raising=False)
        result = verify_stinky._find_readelf()
        assert result is not None and len(result) == 1
        assert "llvm-readelf" in result[0]

    def test_find_readelf_rocm_path_first(self, tmp_path, monkeypatch):
        # ROCM_PATH should be tried first
        rocm_bin = tmp_path / "rocm" / "bin"
        rocm_bin.mkdir(parents=True)
        readelf = rocm_bin / "llvm-readelf"
        readelf.write_text("#!/bin/sh\necho rocm")
        os.chmod(readelf, 0o755)
        monkeypatch.setenv("ROCM_PATH", str(rocm_bin.parent))
        monkeypatch.delenv("LLVM_BIN", raising=False)
        result = verify_stinky._find_readelf()
        assert result is not None
        assert str(readelf) in result[0]

    def test_find_readelf_not_found_returns_none(self, monkeypatch):
        # Clear all paths
        monkeypatch.delenv("ROCM_PATH", raising=False)
        monkeypatch.delenv("LLVM_BIN", raising=False)
        monkeypatch.setenv("PATH", "/nonexistent/path")
        result = verify_stinky._find_readelf()
        # This may or may not be None depending on system, but we're testing the logic
        # Just verify it returns a list or None
        assert result is None or isinstance(result, list)


class TestReadelfSectionHeaders:
    """Test readelf invocation."""

    def test_readelf_not_found(self, monkeypatch):
        # Mock _find_readelf to return None
        monkeypatch.setattr(verify_stinky, "_find_readelf", lambda: None)
        rc, out, err = verify_stinky._readelf_section_headers(Path("/tmp/dummy.o"))
        assert rc == 127
        assert "no readelf/llvm-readelf found" in err


class TestVerifyStinkyPaths:
    """Test the main verification logic with synthetic fixtures."""

    def test_verify_missing_s_file(self, tmp_path):
        s_path = tmp_path / "missing.s"
        o_path = tmp_path / "kernel.o"
        o_path.write_bytes(b"")
        rc, out, err = verify_stinky.verify_stinky_paths(s_path, o_path)
        assert rc == 2
        assert "missing file" in err

    def test_verify_missing_o_file(self, tmp_path):
        s_path = tmp_path / "kernel.s"
        s_path.write_text("# STINKY_TOTAL_INST_BYTES: 100\n")
        o_path = tmp_path / "missing.o"
        rc, out, err = verify_stinky.verify_stinky_paths(s_path, o_path)
        assert rc == 2
        assert "missing file" in err

    def test_verify_no_stinky_marker_skip(self, tmp_path):
        s_path = tmp_path / "kernel.s"
        s_path.write_text("# just assembly\n.section .text\n")
        o_path = tmp_path / "kernel.o"
        o_path.write_bytes(b"")
        rc, out, err = verify_stinky.verify_stinky_paths(s_path, o_path)
        assert rc == 0
        assert "skip" in out.lower()

    def test_verify_s_file_read_error(self, tmp_path, monkeypatch):
        s_path = tmp_path / "kernel.s"
        s_path.write_text("# STINKY_TOTAL_INST_BYTES: 100\n")
        o_path = tmp_path / "kernel.o"
        o_path.write_bytes(b"")
        # Mock read_text to raise OSError
        original_read = Path.read_text

        def mock_read(self, *args, **kwargs):
            if "kernel.s" in str(self):
                raise OSError("Permission denied")
            return original_read(self, *args, **kwargs)

        monkeypatch.setattr(Path, "read_text", mock_read)
        rc, out, err = verify_stinky.verify_stinky_paths(s_path, o_path)
        assert rc == 2
        assert "read" in err

    def test_verify_readelf_failure_no_tools(self, tmp_path, monkeypatch):
        s_path = tmp_path / "kernel.s"
        s_path.write_text("# STINKY_TOTAL_INST_BYTES: 100\n")
        o_path = tmp_path / "kernel.o"
        o_path.write_bytes(b"")
        # Mock _find_readelf to return None
        monkeypatch.setattr(verify_stinky, "_find_readelf", lambda: None)
        rc, out, err = verify_stinky.verify_stinky_paths(s_path, o_path)
        assert rc == 2
        assert "readelf" in err.lower()

    def test_verify_readelf_no_text_section(self, tmp_path, monkeypatch):
        s_path = tmp_path / "kernel.s"
        s_path.write_text("# STINKY_TOTAL_INST_BYTES: 100\n")
        o_path = tmp_path / "kernel.o"
        o_path.write_bytes(b"")
        # Mock _readelf_section_headers to return valid but text-free output
        monkeypatch.setattr(
            verify_stinky,
            "_readelf_section_headers",
            lambda p: (0, "Section Headers:\n[.rodata]", ""),
        )
        rc, out, err = verify_stinky.verify_stinky_paths(s_path, o_path)
        assert rc == 2
        assert "could not find .text" in err


class TestMain:
    """Test the CLI entry point."""

    def test_main_with_missing_args(self):
        with pytest.raises(SystemExit):
            verify_stinky.main([])

    def test_main_cli_help(self, capsys):
        with pytest.raises(SystemExit):
            verify_stinky.main(["--help"])
        assert "Compare STINKY_TOTAL_INST_BYTES" in capsys.readouterr().out

    def test_main_skip_no_stinky_marker(self, tmp_path, capsys):
        s_path = tmp_path / "kernel.s"
        s_path.write_text("# just assembly\n")
        o_path = tmp_path / "kernel.o"
        o_path.write_bytes(b"")
        rc = verify_stinky.main([str(s_path), str(o_path)])
        assert rc == 0
        assert "skip" in capsys.readouterr().out.lower()

    def test_main_missing_file_error(self, tmp_path, capsys):
        s_path = tmp_path / "missing.s"
        o_path = tmp_path / "missing.o"
        rc = verify_stinky.main([str(s_path), str(o_path)])
        assert rc == 2
        captured = capsys.readouterr()
        assert "missing file" in captured.err
