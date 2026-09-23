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
import os
import stat
from pathlib import Path
from unittest.mock import Mock, patch, mock_open, MagicMock, call

from Tensile.TensileBenchmarkClusterScripts import (
    ScriptHelper,
    ScriptWriter,
    BenchmarkNodeWriter,
    BenchmarkTaskWriter,
    BenchmarkJobWriter,
)


@pytest.mark.unit
class TestScriptHelper:
    """Test suite for ScriptHelper class."""

    def test_genLine_simple_text(self):
        """ScriptHelper.genLine generates line with newline."""
        result = ScriptHelper.genLine("test")
        assert result == "test\n"

    def test_genLine_empty_string(self):
        """ScriptHelper.genLine handles empty string."""
        result = ScriptHelper.genLine("")
        assert result == "\n"

    def test_genLine_with_numbers(self):
        """ScriptHelper.genLine handles numeric input."""
        result = ScriptHelper.genLine(123)
        assert result == "123\n"

    def test_genComment_adds_hash(self):
        """ScriptHelper.genComment adds hash prefix."""
        result = ScriptHelper.genComment("test comment")
        assert result == "#test comment\n"

    def test_genComment_empty(self):
        """ScriptHelper.genComment handles empty comment."""
        result = ScriptHelper.genComment("")
        assert result == "#\n"

    def test_genEcho_wraps_in_echo(self):
        """ScriptHelper.genEcho wraps text in echo command."""
        result = ScriptHelper.genEcho("hello")
        assert result == 'echo "hello";\n'

    def test_genEcho_escapes_quotes(self):
        """ScriptHelper.genEcho preserves quotes in text."""
        result = ScriptHelper.genEcho('say "hi"')
        assert 'echo "say "hi""' in result or 'echo "say \\"hi\\""' in result

    @patch("os.chmod")
    @patch("os.stat")
    def test_makeExecutable_sets_permissions(self, mock_stat, mock_chmod):
        """ScriptHelper.makeExecutable sets executable permissions."""
        mock_stat_result = Mock()
        mock_stat_result.st_mode = 0o644
        mock_stat.return_value = mock_stat_result

        ScriptHelper.makeExecutable("/path/to/script.sh")

        mock_stat.assert_called_once_with("/path/to/script.sh")
        mock_chmod.assert_called_once()
        # Check that executable bits were added
        called_mode = mock_chmod.call_args[0][1]
        assert called_mode & 0o111  # At least one execute bit set

    @patch("os.chmod")
    @patch("os.stat")
    def test_makeExecutable_preserves_read_bits(self, mock_stat, mock_chmod):
        """ScriptHelper.makeExecutable preserves read permissions."""
        mock_stat_result = Mock()
        mock_stat_result.st_mode = 0o444  # r--r--r--
        mock_stat.return_value = mock_stat_result

        ScriptHelper.makeExecutable("/path/to/script.sh")

        called_mode = mock_chmod.call_args[0][1]
        # Should be r-xr-xr-x (0o555)
        assert called_mode == 0o555


@pytest.mark.unit
class TestScriptWriter:
    """Test suite for ScriptWriter class."""

    def test_init_creates_empty_buffer(self):
        """ScriptWriter initializes with empty buffer."""
        writer = ScriptWriter()
        assert writer._buffer == ""

    def test_writeLine_adds_line(self):
        """ScriptWriter.writeLine adds line to buffer."""
        writer = ScriptWriter()
        writer.writeLine("test")
        assert writer._buffer == "test\n"

    def test_writeLine_multiple_lines(self):
        """ScriptWriter.writeLine accumulates multiple lines."""
        writer = ScriptWriter()
        writer.writeLine("line1")
        writer.writeLine("line2")
        assert writer._buffer == "line1\nline2\n"

    def test_writeComment_adds_comment(self):
        """ScriptWriter.writeComment adds comment to buffer."""
        writer = ScriptWriter()
        writer.writeComment("comment")
        assert writer._buffer == "#comment\n"

    def test_writeEcho_adds_echo(self):
        """ScriptWriter.writeEcho adds echo command to buffer."""
        writer = ScriptWriter()
        writer.writeEcho("message")
        assert 'echo "message";' in writer._buffer

    def test_str_returns_buffer(self):
        """ScriptWriter.__str__ returns buffer content."""
        writer = ScriptWriter()
        writer.writeLine("test")
        assert str(writer) == "test\n"

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_dumpToFile_writes_to_file(self, mock_exec, mock_file):
        """ScriptWriter.dumpToFile writes buffer to file."""
        writer = ScriptWriter()
        writer.writeLine("#!/bin/bash")
        writer.writeLine("echo 'test'")

        writer.dumpToFile("script.sh", "/output")

        mock_file.assert_called_once_with("/output/script.sh", "w")
        mock_file().write.assert_called_once_with("#!/bin/bash\necho 'test'\n")

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_dumpToFile_makes_executable(self, mock_exec, mock_file):
        """ScriptWriter.dumpToFile makes file executable."""
        writer = ScriptWriter()
        writer.writeLine("test")

        writer.dumpToFile("script.sh", "/output")

        mock_exec.assert_called_once_with("/output/script.sh")

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_dumpToFile_default_output_dir(self, mock_exec, mock_file):
        """ScriptWriter.dumpToFile uses current dir by default."""
        writer = ScriptWriter()
        writer.writeLine("test")

        writer.dumpToFile("script.sh")

        mock_file.assert_called_once_with("./script.sh", "w")

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_writeScript_factory_method(self, mock_exec, mock_file):
        """ScriptWriter.writeScript factory method creates script."""
        mock_writer_class = Mock()

        # Call writeScript - it will use ScriptWriter as the buffer class (cls)
        ScriptWriter.writeScript(mock_writer_class, "test.sh", "/out")

        # Verify the writer methods were called
        mock_writer_class.writeHeader.assert_called_once()
        mock_writer_class.writeBody.assert_called_once()
        mock_writer_class.writeEpilogue.assert_called_once()

        # Verify file was written
        mock_file.assert_called_once_with("/out/test.sh", "w")
        mock_exec.assert_called_once_with("/out/test.sh")

    @patch("builtins.print")
    def test_previewScript_prints_buffer(self, mock_print):
        """ScriptWriter.previewScript prints script content."""
        mock_writer_class = Mock()

        # Call previewScript - it will use ScriptWriter as the buffer class (cls)
        ScriptWriter.previewScript(mock_writer_class)

        # Verify the writer methods were called
        mock_writer_class.writeHeader.assert_called_once()
        mock_writer_class.writeBody.assert_called_once()
        mock_writer_class.writeEpilogue.assert_called_once()

        # Verify print was called
        mock_print.assert_called_once()


@pytest.mark.unit
class TestBenchmarkNodeWriter:
    """Test suite for BenchmarkNodeWriter class."""

    def test_writeHeader_adds_header(self):
        """BenchmarkNodeWriter.writeHeader writes header comments."""
        buffer = ScriptWriter()
        BenchmarkNodeWriter.writeHeader(buffer)

        content = str(buffer)
        assert "#!/bin/bash" in content
        assert "#Benchmark node script" in content

    def test_writeBody_includes_functions(self):
        """BenchmarkNodeWriter.writeBody includes function definitions."""
        buffer = ScriptWriter()
        BenchmarkNodeWriter.writeBody(buffer)

        content = str(buffer)
        assert "usage()" in content
        assert "failAndExit()" in content

    def test_writeBody_includes_arg_parsing(self):
        """BenchmarkNodeWriter.writeBody includes argument parsing."""
        buffer = ScriptWriter()
        BenchmarkNodeWriter.writeBody(buffer)

        content = str(buffer)
        assert "getopts" in content
        assert "dockerImagePath" in content
        assert "logDir" in content
        assert "resultDir" in content
        assert "taskDir" in content

    def test_writeBody_includes_docker_load(self):
        """BenchmarkNodeWriter.writeBody includes docker load commands."""
        buffer = ScriptWriter()
        BenchmarkNodeWriter.writeBody(buffer)

        content = str(buffer)
        assert "docker load" in content
        assert "dockerImageId" in content

    def test_writeBody_includes_docker_run(self):
        """BenchmarkNodeWriter.writeBody includes docker run command."""
        buffer = ScriptWriter()
        BenchmarkNodeWriter.writeBody(buffer)

        content = str(buffer)
        assert "docker run" in content
        assert "--device=/dev/kfd" in content
        assert "--device=/dev/dri" in content

    def test_writeEpilogue_adds_exit(self):
        """BenchmarkNodeWriter.writeEpilogue adds exit statement."""
        buffer = ScriptWriter()
        BenchmarkNodeWriter.writeEpilogue(buffer)

        content = str(buffer)
        assert "exit $returnCode" in content
        assert "Done!" in content

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_full_script_generation(self, mock_exec, mock_file):
        """BenchmarkNodeWriter generates complete script."""
        buffer = ScriptWriter()
        BenchmarkNodeWriter.writeHeader(buffer)
        BenchmarkNodeWriter.writeBody(buffer)
        BenchmarkNodeWriter.writeEpilogue(buffer)

        content = str(buffer)
        # Check structure
        assert content.startswith("#")
        assert "#!/bin/bash" in content or "#!/bin/bash" in content
        assert "docker" in content
        assert "exit" in content


@pytest.mark.unit
class TestBenchmarkTaskWriter:
    """Test suite for BenchmarkTaskWriter class."""

    def test_writeHeader_adds_header(self):
        """BenchmarkTaskWriter.writeHeader writes header comments."""
        buffer = ScriptWriter()
        BenchmarkTaskWriter.writeHeader(buffer)

        content = str(buffer)
        assert "#!/bin/bash" in content
        assert "#Benchmark task enqueue script" in content

    def test_writeBody_includes_functions(self):
        """BenchmarkTaskWriter.writeBody includes function definitions."""
        buffer = ScriptWriter()
        BenchmarkTaskWriter.writeBody(buffer)

        content = str(buffer)
        assert "usage()" in content
        assert "failAndExit()" in content

    def test_writeBody_includes_arg_parsing(self):
        """BenchmarkTaskWriter.writeBody includes argument parsing."""
        buffer = ScriptWriter()
        BenchmarkTaskWriter.writeBody(buffer)

        content = str(buffer)
        assert "getopts" in content
        assert "imageDir" in content
        assert "logsDir" in content
        assert "resultsDir" in content
        assert "tasksDir" in content

    def test_writeBody_includes_task_config(self):
        """BenchmarkTaskWriter.writeBody includes task configuration."""
        buffer = ScriptWriter()
        BenchmarkTaskWriter.writeBody(buffer)

        content = str(buffer)
        assert "SLURM_ARRAY_TASK_ID" in content
        assert "taskName" in content
        assert "mkdir -p" in content

    def test_writeBody_includes_srun(self):
        """BenchmarkTaskWriter.writeBody includes srun command."""
        buffer = ScriptWriter()
        BenchmarkTaskWriter.writeBody(buffer)

        content = str(buffer)
        assert "srun" in content
        assert "-N 1" in content

    def test_writeEpilogue_adds_exit(self):
        """BenchmarkTaskWriter.writeEpilogue adds exit statement."""
        buffer = ScriptWriter()
        BenchmarkTaskWriter.writeEpilogue(buffer)

        content = str(buffer)
        assert "exit $returnCode" in content
        assert "Done!" in content

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_full_script_generation(self, mock_exec, mock_file):
        """BenchmarkTaskWriter generates complete script."""
        buffer = ScriptWriter()
        BenchmarkTaskWriter.writeHeader(buffer)
        BenchmarkTaskWriter.writeBody(buffer)
        BenchmarkTaskWriter.writeEpilogue(buffer)

        content = str(buffer)
        # Check structure
        assert content.startswith("#")
        assert "srun" in content
        assert "SLURM" in content
        assert "exit" in content


@pytest.mark.unit
class TestBenchmarkJobWriter:
    """Test suite for BenchmarkJobWriter class."""

    def test_writeHeader_adds_header(self):
        """BenchmarkJobWriter.writeHeader writes header comments."""
        buffer = ScriptWriter()
        BenchmarkJobWriter.writeHeader(buffer)

        content = str(buffer)
        assert "#!/bin/bash" in content
        assert "#Benchmark batch job script" in content

    def test_writeBody_includes_functions(self):
        """BenchmarkJobWriter.writeBody includes function definitions."""
        buffer = ScriptWriter()
        BenchmarkJobWriter.writeBody(buffer)

        content = str(buffer)
        assert "usage()" in content
        assert "failAndExit()" in content

    def test_writeBody_includes_arg_parsing(self):
        """BenchmarkJobWriter.writeBody includes argument parsing."""
        buffer = ScriptWriter()
        BenchmarkJobWriter.writeBody(buffer)

        content = str(buffer)
        assert "getopts" in content
        assert "imageDir" in content
        assert "logsDir" in content
        assert "resultsDir" in content
        assert "tasksDir" in content
        assert "taskScript" in content

    def test_writeBody_includes_job_config(self):
        """BenchmarkJobWriter.writeBody includes job configuration."""
        buffer = ScriptWriter()
        BenchmarkJobWriter.writeBody(buffer)

        content = str(buffer)
        assert "arraySize" in content
        assert "arrayStart" in content
        assert "arrayEnd" in content
        assert "slurmLogsDir" in content

    def test_writeBody_includes_sbatch(self):
        """BenchmarkJobWriter.writeBody includes sbatch command."""
        buffer = ScriptWriter()
        BenchmarkJobWriter.writeBody(buffer)

        content = str(buffer)
        assert "sbatch" in content
        assert "--nodes=1" in content
        assert "--array=" in content
        assert "--wait" in content

    def test_writeEpilogue_adds_exit(self):
        """BenchmarkJobWriter.writeEpilogue adds exit statement."""
        buffer = ScriptWriter()
        BenchmarkJobWriter.writeEpilogue(buffer)

        content = str(buffer)
        assert "exit $returnCode" in content
        assert "Done!" in content

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_full_script_generation(self, mock_exec, mock_file):
        """BenchmarkJobWriter generates complete script."""
        buffer = ScriptWriter()
        BenchmarkJobWriter.writeHeader(buffer)
        BenchmarkJobWriter.writeBody(buffer)
        BenchmarkJobWriter.writeEpilogue(buffer)

        content = str(buffer)
        # Check structure
        assert content.startswith("#")
        assert "sbatch" in content
        assert "SLURM" in content
        assert "exit" in content


@pytest.mark.unit
class TestScriptWriterFactoryMethods:
    """Test suite for ScriptWriter factory methods."""

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_writeBenchmarkNodeScript_creates_file(self, mock_exec, mock_file):
        """ScriptWriter.writeBenchmarkNodeScript creates node script file."""
        ScriptWriter.writeBenchmarkNodeScript("node.sh", "/output")

        mock_file.assert_called_once_with("/output/node.sh", "w")
        mock_exec.assert_called_once_with("/output/node.sh")

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_writeBenchmarkTaskScript_creates_file(self, mock_exec, mock_file):
        """ScriptWriter.writeBenchmarkTaskScript creates task script file."""
        ScriptWriter.writeBenchmarkTaskScript("task.sh", "/output")

        mock_file.assert_called_once_with("/output/task.sh", "w")
        mock_exec.assert_called_once_with("/output/task.sh")

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_writeBenchmarkJobScript_creates_file(self, mock_exec, mock_file):
        """ScriptWriter.writeBenchmarkJobScript creates job script file."""
        ScriptWriter.writeBenchmarkJobScript("job.sh", "/output")

        mock_file.assert_called_once_with("/output/job.sh", "w")
        mock_exec.assert_called_once_with("/output/job.sh")

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_writeBenchmarkNodeScript_content(self, mock_exec, mock_file):
        """ScriptWriter.writeBenchmarkNodeScript writes docker-related content."""
        ScriptWriter.writeBenchmarkNodeScript("node.sh", "/output")

        written_content = mock_file().write.call_args[0][0]
        assert "docker" in written_content
        assert "usage" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_writeBenchmarkTaskScript_content(self, mock_exec, mock_file):
        """ScriptWriter.writeBenchmarkTaskScript writes SLURM srun content."""
        ScriptWriter.writeBenchmarkTaskScript("task.sh", "/output")

        written_content = mock_file().write.call_args[0][0]
        assert "srun" in written_content
        assert "SLURM" in written_content

    @patch("builtins.open", new_callable=mock_open)
    @patch("Tensile.TensileBenchmarkClusterScripts.ScriptHelper.makeExecutable")
    def test_writeBenchmarkJobScript_content(self, mock_exec, mock_file):
        """ScriptWriter.writeBenchmarkJobScript writes SLURM sbatch content."""
        ScriptWriter.writeBenchmarkJobScript("job.sh", "/output")

        written_content = mock_file().write.call_args[0][0]
        assert "sbatch" in written_content
        assert "array" in written_content
