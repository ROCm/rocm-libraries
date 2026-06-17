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
import sys
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock, call, mock_open

from Tensile.TensileBenchmarkCluster import (
    BenchmarkImplSLURM,
    TensileBenchmarkCluster,
)


@pytest.mark.unit
class TestBenchmarkImplSLURM:
    """Test suite for BenchmarkImplSLURM class."""

    def test_initialize_config_creates_slurm_section(self):
        """BenchmarkImplSLURM.initializeConfig creates SLURM configuration section."""
        mock_config = MagicMock()
        mock_config.__getitem__ = Mock(return_value="/fake/tensile/dir")
        mock_slurm_section = MagicMock()
        mock_config.createSection = Mock(return_value=mock_slurm_section)

        BenchmarkImplSLURM.initializeConfig(mock_config)

        mock_config.createSection.assert_called_with("SLURM")

    def test_initialize_config_creates_scripts_section(self):
        """BenchmarkImplSLURM.initializeConfig creates SCRIPTS subsection."""
        mock_config = MagicMock()
        mock_config.__getitem__ = Mock(return_value="/fake/tensile/dir")
        mock_slurm_section = MagicMock()
        mock_scripts_section = MagicMock()
        mock_config.createSection = Mock(return_value=mock_slurm_section)
        mock_slurm_section.createSection = Mock(return_value=mock_scripts_section)

        BenchmarkImplSLURM.initializeConfig(mock_config)

        # Should create both SCRIPTS and DOCKER sections
        assert mock_slurm_section.createSection.call_count == 2
        calls = [str(c) for c in mock_slurm_section.createSection.call_args_list]
        assert any("SCRIPTS" in str(c) for c in calls)
        assert any("DOCKER" in str(c) for c in calls)

    def test_initialize_config_sets_script_names(self):
        """BenchmarkImplSLURM.initializeConfig sets default script names."""
        mock_config = MagicMock()
        mock_config.__getitem__ = Mock(return_value="/fake/tensile/dir")
        mock_slurm_section = MagicMock()
        mock_scripts_section = MagicMock()
        mock_config.createSection = Mock(return_value=mock_slurm_section)
        mock_slurm_section.createSection = Mock(return_value=mock_scripts_section)

        BenchmarkImplSLURM.initializeConfig(mock_config)

        # Check that script names are set
        script_calls = mock_scripts_section.createValue.call_args_list
        assert any("JobScriptName" in str(c) and "runBenchmark.sh" in str(c) for c in script_calls)
        assert any("TaskScriptName" in str(c) and "enqueueTask.sh" in str(c) for c in script_calls)

    def test_initialize_config_sets_docker_values(self):
        """BenchmarkImplSLURM.initializeConfig sets Docker configuration values."""
        mock_config = MagicMock()
        mock_config.__getitem__ = Mock(return_value="/fake/tensile/dir")
        mock_slurm_section = MagicMock()
        mock_docker_section = MagicMock()

        # Return DOCKER section on second createSection call
        mock_slurm_section.createSection = Mock(side_effect=[MagicMock(), mock_docker_section])
        mock_config.createSection = Mock(return_value=mock_slurm_section)

        BenchmarkImplSLURM.initializeConfig(mock_config)

        # Check that Docker values are set
        docker_calls = mock_docker_section.createValue.call_args_list
        assert len(docker_calls) > 0
        assert any("DockerImageName" in str(c) for c in docker_calls)
        assert any("TensileFork" in str(c) for c in docker_calls)

    @patch("os.listdir")
    @patch("Tensile.TensileBenchmarkCluster.ScriptWriter")
    def test_generate_benchmark_creates_scripts(self, mock_script_writer, mock_listdir):
        """BenchmarkImplSLURM.generateBenchmark creates cluster scripts."""
        # Mock os.listdir to return empty list (no config files in tasks dir)
        mock_listdir.return_value = []

        mock_config = {
            "BenchmarkBaseDir": "/fake/base",
            "BenchmarkTasksDir": "/fake/tasks",
            "BenchmarkImageDir": "/fake/image",
            "BenchmarkLogsDir": "/fake/logs",
            "SLURM": {
                "DOCKER": {
                    "DockerBaseImage": "fake:image",
                    "DockerBuildFile": "/fake/dockerfile",
                    "DockerImageName": "tensile-test",
                    "DockerImageTag": "TEST",
                    "TensileFork": "test-fork",
                    "TensileBranch": "test-branch",
                    "TensileCommit": "abc123",
                },
                "SCRIPTS": {
                    "TaskScriptName": "task.sh",
                    "JobScriptName": "job.sh",
                }
            }
        }

        with patch.object(BenchmarkImplSLURM, '_BenchmarkImplSLURM__createTensileBenchmarkContainer'):
            BenchmarkImplSLURM.generateBenchmark(mock_config)

        mock_script_writer.writeBenchmarkJobScript.assert_called_once()
        mock_script_writer.writeBenchmarkTaskScript.assert_called_once()

    @patch("os.rename")
    @patch("os.makedirs")
    @patch("os.path.isfile")
    @patch("os.listdir")
    @patch("Tensile.TensileBenchmarkCluster.ScriptWriter")
    def test_generate_benchmark_processes_config_files(self, mock_script_writer, mock_listdir,
                                                       mock_isfile, mock_makedirs, mock_rename):
        """BenchmarkImplSLURM.generateBenchmark processes config files in tasks directory."""
        # Mock os.listdir to return config files
        mock_listdir.return_value = ["config_0001.yaml", "config_0002.yaml"]
        mock_isfile.return_value = True

        mock_config = {
            "BenchmarkBaseDir": "/fake/base",
            "BenchmarkTasksDir": "/fake/tasks",
            "BenchmarkImageDir": "/fake/image",
            "BenchmarkLogsDir": "/fake/logs",
            "SLURM": {
                "DOCKER": {
                    "DockerBaseImage": "fake:image",
                    "DockerBuildFile": "/fake/dockerfile",
                    "DockerImageName": "tensile-test",
                    "DockerImageTag": "TEST",
                    "TensileFork": "test-fork",
                    "TensileBranch": "test-branch",
                    "TensileCommit": "abc123",
                },
                "SCRIPTS": {
                    "TaskScriptName": "task.sh",
                    "JobScriptName": "job.sh",
                }
            }
        }

        with patch.object(BenchmarkImplSLURM, '_BenchmarkImplSLURM__createTensileBenchmarkContainer'):
            BenchmarkImplSLURM.generateBenchmark(mock_config)

        # Should create subdirectories for each config file
        assert mock_makedirs.call_count == 2
        # Should rename/move config files into subdirectories
        assert mock_rename.call_count == 2
        # Should create node scripts for each config
        assert mock_script_writer.writeBenchmarkNodeScript.call_count == 2

    @patch("subprocess.check_call")
    @patch("builtins.open", new_callable=mock_open)
    def test_invoke_benchmark_runs_script(self, mock_file, mock_subprocess):
        """BenchmarkImplSLURM.invokeBenchmark runs the job script."""
        mock_config = {
            "BenchmarkBaseDir": "/fake/base",
            "BenchmarkTasksDir": "/fake/tasks",
            "BenchmarkImageDir": "/fake/image",
            "BenchmarkResultsDir": "/fake/results",
            "BenchmarkLogsDir": "/fake/logs",
            "SLURM": {
                "SCRIPTS": {
                    "TaskScriptName": "task.sh",
                    "JobScriptName": "job.sh",
                }
            }
        }

        BenchmarkImplSLURM.invokeBenchmark(mock_config)

        # Should open log file
        mock_file.assert_called_once()
        assert "job.log" in str(mock_file.call_args)
        # Should call subprocess to run script
        mock_subprocess.assert_called_once()

    def test_pre_invoke_benchmark_does_nothing(self):
        """BenchmarkImplSLURM.preInvokeBenchmark is a no-op."""
        result = BenchmarkImplSLURM.preInvokeBenchmark({})
        assert result is None

    def test_post_invoke_benchmark_does_nothing(self):
        """BenchmarkImplSLURM.postInvokeBenchmark is a no-op."""
        result = BenchmarkImplSLURM.postInvokeBenchmark({})
        assert result is None


@pytest.mark.unit
class TestTensileBenchmarkCluster:
    """Test suite for TensileBenchmarkCluster class."""

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_init_creates_config(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.__init__ creates configuration."""
        args = ["logic.yaml", "/deploy/path"]
        cluster = TensileBenchmarkCluster(args)

        mock_project_config.assert_called_once()
        assert cluster._config is not None

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_init_with_slurm_backend(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster uses SLURM backend by default."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)

        assert cluster._backendImpl == BenchmarkImplSLURM

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path", "--cluster-backend", "unknown"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_init_raises_on_unknown_backend(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster raises on unknown backend."""
        args = ["logic.yaml", "/deploy/path", "--cluster-backend", "unknown"]

        mock_config_instance = MagicMock()
        mock_project_config.return_value = mock_config_instance

        with pytest.raises(NotImplementedError, match="Cluster backend not recognized"):
            TensileBenchmarkCluster(args)

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_workflow_steps_returns_tuple(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.workflowSteps returns tuple of workflow flags."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(side_effect=lambda k: {
            "RunDeployStep": True,
            "RunBenchmarkStep": False,
            "RunResultsStep": True,
        }.get(k))
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        steps = cluster.workflowSteps()

        assert steps == (True, False, True)

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_config_returns_config_object(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.config returns configuration object."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        config = cluster.config()

        assert config == mock_config_instance

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_base_dir_returns_base_directory(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.baseDir returns base directory from config."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(return_value="/fake/base/dir")
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        base_dir = cluster.baseDir()

        assert base_dir == "/fake/base/dir"
        mock_config_instance.__getitem__.assert_called_with("BenchmarkBaseDir")

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_tasks_dir_returns_tasks_directory(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.tasksDir returns tasks directory."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(return_value="/fake/tasks/dir")
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        tasks_dir = cluster.tasksDir()

        assert tasks_dir == "/fake/tasks/dir"
        mock_config_instance.__getitem__.assert_called_with("BenchmarkTasksDir")

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_image_dir_returns_image_directory(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.imageDir returns image directory."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(return_value="/fake/image/dir")
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        image_dir = cluster.imageDir()

        assert image_dir == "/fake/image/dir"

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_results_dir_returns_results_directory(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.resultsDir returns results directory."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(return_value="/fake/results/dir")
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        results_dir = cluster.resultsDir()

        assert results_dir == "/fake/results/dir"

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_final_logic_dir_returns_final_logic_directory(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.finalLogicDir returns final logic directory."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(return_value="/fake/final/dir")
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        final_dir = cluster.finalLogicDir()

        assert final_dir == "/fake/final/dir"

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_logs_dir_returns_logs_directory(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.logsDir returns logs directory."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(return_value="/fake/logs/dir")
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        logs_dir = cluster.logsDir()

        assert logs_dir == "/fake/logs/dir"

    @patch("os.makedirs")
    def test_ensure_path_creates_directory(self, mock_makedirs):
        """TensileBenchmarkCluster.ensurePath creates directory."""
        path = "/fake/new/path"
        result = TensileBenchmarkCluster.ensurePath(path)

        mock_makedirs.assert_called_once_with(path)
        assert result == path

    @patch("os.makedirs", side_effect=OSError("Directory exists"))
    def test_ensure_path_ignores_os_error(self, mock_makedirs):
        """TensileBenchmarkCluster.ensurePath ignores OSError."""
        path = "/existing/path"
        result = TensileBenchmarkCluster.ensurePath(path)

        mock_makedirs.assert_called_once_with(path)
        assert result == path  # Should still return path even on error

    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_root_tensile_dir_returns_root_directory(self, mock_project_config, mock_init):
        """TensileBenchmarkCluster.rootTensileDir returns root Tensile directory."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(return_value="/fake/root/tensile")
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        root_dir = cluster.rootTensileDir()

        assert root_dir == "/fake/root/tensile"
        mock_config_instance.__getitem__.assert_called_with("RootTensileDir")

    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkSplitter")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.generateBenchmark")
    @patch("os.makedirs")
    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path", "--deploy-only"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_generate_cluster_benchmark_creates_directories(self, mock_project_config, mock_init,
                                                           mock_makedirs, mock_generate, mock_splitter,
                                                           mock_listdir, mock_isdir):
        """TensileBenchmarkCluster generate workflow creates necessary directories."""
        args = ["logic.yaml", "/deploy/path", "--deploy-only"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(side_effect=lambda k: {
            "BenchmarkBaseDir": "/fake/base",
            "BenchmarkTasksDir": "/fake/tasks",
            "BenchmarkImageDir": "/fake/image",
            "BenchmarkResultsDir": "/fake/results",
            "BenchmarkFinalLogicDir": "/fake/final",
            "BenchmarkLogsDir": "/fake/logs",
            "BenchmarkLogicPath": "logic.yaml",
            "BenchmarkTaskSize": 10,
            "RunDeployStep": True,
            "RunBenchmarkStep": False,
            "RunResultsStep": False,
        }.get(k))
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        cluster.main()

        # Should create all necessary directories
        assert mock_makedirs.call_count >= 6
        # Should split benchmark
        mock_splitter.splitBenchmarkBySizes.assert_called_once()
        # Should generate benchmark
        mock_generate.assert_called_once()

    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.postInvokeBenchmark")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.invokeBenchmark")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.preInvokeBenchmark")
    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path", "--run-only"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_run_cluster_benchmark_invokes_backend(self, mock_project_config, mock_init,
                                                   mock_pre, mock_invoke, mock_post):
        """TensileBenchmarkCluster run workflow invokes backend methods."""
        args = ["logic.yaml", "/deploy/path", "--run-only"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(side_effect=lambda k: {
            "RunDeployStep": False,
            "RunBenchmarkStep": True,
            "RunResultsStep": False,
        }.get(k))
        mock_project_config.return_value = mock_config_instance

        cluster = TensileBenchmarkCluster(args)
        cluster.main()

        # Should call all three backend methods in order
        mock_pre.assert_called_once()
        mock_invoke.assert_called_once()
        mock_post.assert_called_once()

    @patch("Tensile.TensileBenchmarkCluster.mergePartialLogics")
    @patch("os.path.isfile")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path", "--results-only"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_combine_results_merges_partial_logics(self, mock_project_config, mock_init,
                                                   mock_listdir, mock_isdir, mock_isfile, mock_merge):
        """TensileBenchmarkCluster combine workflow merges partial results."""
        args = ["logic.yaml", "/deploy/path", "--results-only"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(side_effect=lambda k: {
            "BenchmarkResultsDir": "/fake/results",
            "BenchmarkFinalLogicDir": "/fake/final",
            "FinalLogicForceMerge": False,
            "FinalLogicTrim": True,
            "RunDeployStep": False,
            "RunBenchmarkStep": False,
            "RunResultsStep": True,
        }.get(k))
        mock_project_config.return_value = mock_config_instance

        # Mock directory structure with results
        mock_listdir.side_effect = [
            ["result_0001", "result_0002"],  # First call: list results subdirs
            ["logic.yaml"],  # Second call: files in first subdir
            ["logic.yaml"],  # Third call: files in second subdir
        ]
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        cluster = TensileBenchmarkCluster(args)
        cluster.main()

        # Should merge partial logics
        mock_merge.assert_called_once()

    @patch("Tensile.TensileBenchmarkCluster.mergePartialLogics")
    @patch("os.path.isfile")
    @patch("os.path.isdir")
    @patch("os.listdir")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.postInvokeBenchmark")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.invokeBenchmark")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.preInvokeBenchmark")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.generateBenchmark")
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkSplitter")
    @patch("os.makedirs")
    @patch("sys.argv", ["script.py", "logic.yaml", "/deploy/path"])
    @patch("Tensile.TensileBenchmarkCluster.BenchmarkImplSLURM.initializeConfig")
    @patch("Tensile.TensileBenchmarkCluster.ProjectConfig")
    def test_main_runs_all_workflow_steps(self, mock_project_config, mock_init,
                                         mock_makedirs, mock_splitter, mock_generate,
                                         mock_pre, mock_invoke, mock_post,
                                         mock_listdir, mock_isdir, mock_isfile, mock_merge):
        """TensileBenchmarkCluster.main runs all workflow steps."""
        args = ["logic.yaml", "/deploy/path"]

        mock_config_instance = MagicMock()
        mock_config_instance.__getitem__ = Mock(side_effect=lambda k: {
            "RunDeployStep": True,
            "RunBenchmarkStep": True,
            "RunResultsStep": True,
            "BenchmarkBaseDir": "/fake/base",
            "BenchmarkTasksDir": "/fake/tasks",
            "BenchmarkImageDir": "/fake/image",
            "BenchmarkResultsDir": "/fake/results",
            "BenchmarkFinalLogicDir": "/fake/final",
            "BenchmarkLogsDir": "/fake/logs",
            "BenchmarkLogicPath": "logic.yaml",
            "BenchmarkTaskSize": 10,
            "FinalLogicForceMerge": False,
            "FinalLogicTrim": True,
        }.get(k))
        mock_project_config.return_value = mock_config_instance

        # Mock directory structure
        mock_listdir.side_effect = [
            ["result_0001"],
            ["logic.yaml"],
        ]
        mock_isdir.return_value = True
        mock_isfile.return_value = True

        cluster = TensileBenchmarkCluster(args)
        cluster.main()

        # Should run all three steps
        mock_generate.assert_called_once()
        mock_invoke.assert_called_once()
        mock_merge.assert_called_once()
