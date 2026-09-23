################################################################################
#
# Copyright (C) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# SPDX-License-Identifier: MIT
################################################################################

import pytest
from io import StringIO
from pathlib import Path
from unittest.mock import Mock, patch, mock_open

from Tensile.EmbeddedData import (
    Namespace,
    Indent,
    EmbeddedDataFile,
)


@pytest.mark.unit
class TestNamespace:
    """Test suite for Namespace context manager."""

    def test_init_stores_parent(self):
        """Constructor stores parent reference."""
        parent = Mock()
        ns = Namespace(parent, "test")
        assert ns.parent is parent

    def test_init_stores_name(self):
        """Constructor stores namespace name."""
        parent = Mock()
        ns = Namespace(parent, "MyNamespace")
        assert ns.name == "MyNamespace"

    def test_init_stores_none_name(self):
        """Constructor stores None for anonymous namespace."""
        parent = Mock()
        ns = Namespace(parent, None)
        assert ns.name is None

    def test_enter_writes_named_namespace(self):
        """__enter__ writes namespace declaration with name."""
        parent = Mock()
        ns = Namespace(parent, "TestNS")

        result = ns.__enter__()

        parent.write.assert_called_once_with("namespace TestNS {")
        parent.indent.assert_called_once()
        assert result is ns

    def test_enter_writes_anonymous_namespace(self):
        """__enter__ writes anonymous namespace declaration."""
        parent = Mock()
        ns = Namespace(parent, None)

        ns.__enter__()

        parent.write.assert_called_once_with("namespace {")
        parent.indent.assert_called_once()

    def test_exit_writes_named_namespace_close(self):
        """__exit__ writes closing brace with namespace name."""
        parent = Mock()
        ns = Namespace(parent, "TestNS")

        ns.__exit__(None, None, None)

        parent.dedent.assert_called_once()
        parent.write.assert_called_once_with("} // namespace TestNS")

    def test_exit_writes_anonymous_namespace_close(self):
        """__exit__ writes closing brace for anonymous namespace."""
        parent = Mock()
        ns = Namespace(parent, None)

        ns.__exit__(None, None, None)

        parent.dedent.assert_called_once()
        parent.write.assert_called_once_with("} // anonymous namespace")

    def test_context_manager_usage(self):
        """Namespace works as context manager."""
        parent = Mock()

        with Namespace(parent, "Test") as ns:
            assert isinstance(ns, Namespace)

        assert parent.write.call_count == 2
        assert parent.indent.call_count == 1
        assert parent.dedent.call_count == 1


@pytest.mark.unit
class TestIndent:
    """Test suite for Indent context manager."""

    def test_init_stores_parent(self):
        """Constructor stores parent reference."""
        parent = Mock()
        ind = Indent(parent)
        assert ind.parent is parent

    def test_enter_returns_self(self):
        """__enter__ returns self."""
        parent = Mock()
        ind = Indent(parent)
        result = ind.__enter__()
        assert result is ind

    def test_exit_calls_dedent(self):
        """__exit__ calls parent.dedent()."""
        parent = Mock()
        ind = Indent(parent)

        ind.__exit__(None, None, None)

        parent.dedent.assert_called_once()

    def test_context_manager_usage(self):
        """Indent works as context manager."""
        parent = Mock()

        with Indent(parent) as ind:
            assert isinstance(ind, Indent)

        parent.dedent.assert_called_once()


@pytest.mark.unit
class TestEmbeddedDataFile:
    """Test suite for EmbeddedDataFile class."""

    def test_init_stores_filename(self):
        """Constructor stores filename."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        assert edf.filename == "test.hpp"

    def test_init_stores_indent_spaces(self):
        """Constructor stores indent_spaces."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file, indent_spaces=2)
        assert edf._indent_spaces == 2

    def test_init_default_indent_spaces(self):
        """Constructor defaults to 4 indent spaces."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        assert edf._indent_spaces == 4

    def test_init_uses_provided_file(self):
        """Constructor uses provided file object."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        assert edf.file is mock_file

    @patch("builtins.open", new_callable=mock_open)
    def test_init_opens_file_when_not_provided(self, mock_file_open):
        """Constructor opens file when file object not provided."""
        edf = EmbeddedDataFile("output.hpp")
        mock_file_open.assert_called_once_with("output.hpp", 'w')

    def test_init_writes_header(self):
        """Constructor writes header including CHeader and includes."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        output = mock_file.getvalue()

        assert "Copyright" in output
        assert "#include <Tensile/EmbeddedData.hpp>" in output
        assert "namespace Tensile {" in output

    def test_include_guard_property(self):
        """include_guard property returns pragma once."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        assert edf.include_guard == "#pragma once"

    def test_includes_property(self):
        """includes property returns C++ include statements."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        includes = edf.includes

        assert "#include <Tensile/EmbeddedData.hpp>" in includes
        assert "#include <Tensile/Contractions.hpp>" in includes
        assert "#include <Tensile/Tensile.hpp>" in includes

    def test_get_lines_from_string(self):
        """get_lines splits string by newlines."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        result = edf.get_lines("line1\nline2\nline3")
        assert result == ["line1", "line2", "line3"]

    def test_get_lines_from_iterable(self):
        """get_lines returns iterable as-is."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        lines = ["line1", "line2"]
        result = edf.get_lines(lines)
        assert result == lines

    def test_get_lines_from_other_type(self):
        """get_lines converts other types to string and splits."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        result = edf.get_lines(42)
        assert result == ["42"]

    def test_format_single_line(self):
        """format applies indentation to single line."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [8]

        result = edf.format("test")
        assert result == "        test\n"

    def test_format_multiple_lines(self):
        """format applies indentation to multiple lines."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [4]

        result = edf.format("line1\nline2")
        assert result == "    line1\n    line2\n"

    def test_indent_level_property_default(self):
        """indent_level returns 0 when no indents."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = []
        assert edf.indent_level == 0

    def test_indent_level_property_with_indents(self):
        """indent_level returns last indent level."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0, 4, 8]
        assert edf.indent_level == 8

    def test_apply_indent_to_none(self):
        """apply_indent with None returns spaces only."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [6]

        result = edf.apply_indent(None)
        assert result == "      "

    def test_apply_indent_strips_and_indents(self):
        """apply_indent strips line and adds indentation."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [4]

        result = edf.apply_indent("  test  ")
        assert result == "    test"

    def test_apply_indent_preserves_preprocessor(self):
        """apply_indent preserves preprocessor directives without indent."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [8]

        result = edf.apply_indent("#include <header>")
        assert result == "#include <header>"

    def test_indent_increases_level_by_default(self):
        """indent() increases indent level by _indent_spaces."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0]
        edf._indent_spaces = 4

        indent_obj = edf.indent()

        assert edf._indent_levels == [0, 4]
        assert isinstance(indent_obj, Indent)

    def test_indent_increases_level_by_custom_amount(self):
        """indent(spaces) increases indent level by specified amount."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0]

        edf.indent(spaces=8)

        assert edf._indent_levels == [0, 8]

    def test_indent_stacks_levels(self):
        """indent() stacks on top of current level."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [4]
        edf._indent_spaces = 4

        edf.indent()

        assert edf._indent_levels == [4, 8]

    def test_dedent_removes_level(self):
        """dedent() removes last indent level."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0, 4, 8]

        edf.dedent()

        assert edf._indent_levels == [0, 4]

    def test_write_single_item(self):
        """write() writes single formatted item to file."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0]

        edf.write("test line")

        output = mock_file.getvalue()
        assert "test line\n" in output

    def test_write_multiple_items(self):
        """write() writes multiple formatted items to file."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0]

        edf.write("line1", "line2", "line3")

        output = mock_file.getvalue()
        assert "line1\n" in output
        assert "line2\n" in output
        assert "line3\n" in output

    def test_comment_writes_cpp_comment(self):
        """comment() writes C++ style comment."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0]

        edf.comment("This is a comment")

        output = mock_file.getvalue()
        assert "// This is a comment\n" in output

    def test_comment_handles_multiline(self):
        """comment() handles multiline text."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._indent_levels = [0]

        edf.comment("Line 1\nLine 2")

        output = mock_file.getvalue()
        assert "// Line 1\n" in output
        assert "// Line 2\n" in output

    def test_namespace_creates_namespace_block(self):
        """namespace() creates and enters namespace."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.namespace("Test")

        assert len(edf._open_blocks) == 2  # Tensile from header + Test
        assert isinstance(edf._open_blocks[-1], Namespace)
        assert edf._open_blocks[-1].name == "Test"

    def test_namespace_creates_anonymous_namespace(self):
        """namespace(None) creates anonymous namespace."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.namespace(None)

        assert len(edf._open_blocks) == 2  # Tensile from header + anonymous
        assert edf._open_blocks[-1].name is None

    def test_end_namespace_closes_matching_namespace(self):
        """end_namespace() closes matching namespace."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf.namespace("Test")
        initial_blocks = len(edf._open_blocks)

        edf.end_namespace("Test")

        assert len(edf._open_blocks) == initial_blocks - 1

    def test_end_namespace_raises_on_type_mismatch(self):
        """end_namespace() raises when top block is not Namespace."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf._open_blocks.append(Indent(edf))

        with pytest.raises(RuntimeError, match="expected Namespace"):
            edf.end_namespace("Test")

    def test_end_namespace_raises_on_name_mismatch(self):
        """end_namespace() raises when namespace name doesn't match."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf.namespace("Actual")

        with pytest.raises(RuntimeError, match="expected Expected, found Actual"):
            edf.end_namespace("Expected")

    def test_embed_data_empty_without_key(self):
        """embed_data() handles empty data without key."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.embed_data("int", [])

        output = mock_file.getvalue()
        assert "EmbedData<int> TENSILE_EMBED_SYMBOL_NAME{};" in output

    def test_embed_data_empty_with_key(self):
        """embed_data() handles empty data with key."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.embed_data("char", [], key="mykey")

        output = mock_file.getvalue()
        assert 'EmbedData<char> TENSILE_EMBED_SYMBOL_NAME("mykey", {});' in output

    def test_embed_data_single_byte_without_key(self):
        """embed_data() embeds single byte without key."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.embed_data("uint8_t", [0x42])

        output = mock_file.getvalue()
        assert "EmbedData<uint8_t> TENSILE_EMBED_SYMBOL_NAME({" in output
        assert "0x42});" in output

    def test_embed_data_single_byte_with_key(self):
        """embed_data() embeds single byte with key."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.embed_data("uint8_t", [0xFF], key="data")

        output = mock_file.getvalue()
        assert 'EmbedData<uint8_t> TENSILE_EMBED_SYMBOL_NAME("data", {' in output
        assert "0xff});" in output

    def test_embed_data_multiple_bytes(self):
        """embed_data() embeds multiple bytes on one line."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.embed_data("uint8_t", [0x01, 0x02, 0x03, 0x04])

        output = mock_file.getvalue()
        assert "0x01, 0x02, 0x03, 0x04});" in output

    def test_embed_data_wraps_at_16_bytes(self):
        """embed_data() wraps lines at 16 bytes."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        data = list(range(20))
        edf.embed_data("uint8_t", data)

        output = mock_file.getvalue()
        lines = output.split('\n')
        # Should have multiple lines with commas
        comma_lines = [l for l in lines if l.strip().endswith(',')]
        assert len(comma_lines) > 0

    def test_embed_data_null_terminated(self):
        """embed_data() appends null byte when nullTerminated=True."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.embed_data("char", [0x41, 0x42], nullTerminated=True)

        output = mock_file.getvalue()
        assert "0x41" in output
        assert "0x42" in output
        assert "0x00" in output

    def test_embed_data_with_comment(self):
        """embed_data() includes comment when provided."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)

        edf.embed_data("int", [1, 2], comment="Test data")

        output = mock_file.getvalue()
        assert "// Test data" in output

    @patch("builtins.open", new_callable=mock_open, read_data=b"\x01\x02\x03")
    def test_embed_file_reads_and_embeds(self, mock_file_open):
        """embed_file() reads file and embeds its contents."""
        output_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=output_file)

        edf.embed_file("uint8_t", "/path/to/data.bin")

        mock_file_open.assert_called_with("/path/to/data.bin", 'rb')
        output = output_file.getvalue()
        assert "0x01" in output
        assert "0x02" in output
        assert "0x03" in output

    @patch("builtins.open", new_callable=mock_open, read_data=b"\xFF")
    def test_embed_file_uses_basename_as_comment(self, mock_file_open):
        """embed_file() uses basename as comment."""
        output_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=output_file)

        edf.embed_file("uint8_t", "/path/to/myfile.dat")

        output = output_file.getvalue()
        assert "// myfile.dat" in output

    @patch("builtins.open", new_callable=mock_open, read_data=b"ABC")
    def test_embed_file_with_key(self, mock_file_open):
        """embed_file() passes key to embed_data."""
        output_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=output_file)

        edf.embed_file("char", "data.txt", key="mydata")

        output = output_file.getvalue()
        assert '"mydata"' in output

    @patch("builtins.open", new_callable=mock_open, read_data=b"test")
    def test_embed_file_null_terminated(self, mock_file_open):
        """embed_file() supports null termination."""
        output_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=output_file)

        edf.embed_file("char", "string.txt", nullTerminated=True)

        output = output_file.getvalue()
        assert "0x00" in output

    def test_exit_closes_all_open_blocks(self):
        """__exit__ closes all remaining open blocks."""
        mock_file = StringIO()
        edf = EmbeddedDataFile("test.hpp", file=mock_file)
        edf.namespace("NS1")
        edf.namespace("NS2")

        # Should have 3 open blocks: Tensile (from header) + NS1 + NS2
        assert len(edf._open_blocks) == 3

        # Get output before __exit__ closes the file
        output_before = mock_file.getvalue()

        edf.__exit__(None, None, None)

        # All blocks should be closed
        assert len(edf._open_blocks) == 0
        # File is closed by __exit__, but we can check the StringIO's buffer
        # by checking what was written before it was closed
        assert mock_file.closed is True

    def test_context_manager_usage(self, tmp_path):
        """EmbeddedDataFile works as context manager."""
        output_file = tmp_path / "test.hpp"

        with EmbeddedDataFile(str(output_file)) as edf:
            edf.write("test content")

        content = output_file.read_text()
        assert "test content" in content
        assert "namespace Tensile" in content
