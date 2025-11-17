#!/usr/bin/env python3
"""
Execution time test executor (Test A)
"""

import subprocess
from pathlib import Path
from typing import Dict, Tuple
from .test_executor import TestExecutor
from .utils import log_info, log_warning, build_test_env_vars


class ExeTimeTest(TestExecutor):
    """Executor for execution time tests"""

    def __init__(self, config: Dict):
        super().__init__(config, 'exe_time')

        # Get build paths
        self.build_dir, self.tensile_sh, self.client_bin = \
            self.build_system.get_build_paths('rel_build')

    def run_single_test(self, test_entry: Tuple[str, str, str]) -> bool:
        """Run a single execution time test"""
        source_yaml, active_modifier, all_modifiers = test_entry

        # Setup test paths
        paths = self.setup_test_paths(source_yaml, active_modifier, all_modifiers)
        yaml_name = paths['yaml_name']
        patched_yaml = paths['patched_yaml']
        log_file = paths['log_file']
        out_dir = paths['out_dir']
        docker_work_dir = paths['docker_work_dir']
        docker_yaml_file = paths['docker_yaml_file']
        test_log_dir = paths['test_log_dir']

        # Clean up old output directory if exists (unless --preserve-build is set)
        if not self.preserve_build:
            self.docker.remove_files(out_dir, docker_work_dir)
        else:
            if self.verbose:
                log_info(f"Preserving existing build directory: {out_dir}")

        # Build environment variables
        env_vars = build_test_env_vars(self.gpu_arch, include_stinkytofu_dump=True)

        # Build environment setup string
        env_setup = ' && '.join([f'export {k}={v}' for k, v in env_vars.items()])
        if env_setup:
            env_setup += ' && '

        # Build the command
        cmd = f"docker exec -w \"{docker_work_dir}\" -it \"{self.container_name}\" bash -c \"{env_setup}{self.tensile_sh} {docker_yaml_file} {out_dir} --prebuilt-client={self.client_bin}\""

        # Execute the command
        if not self.execute_test_command(cmd, log_file, yaml_name):
            return False

        # Copy assembly files from output directory
        if self.verbose:
            log_info("Collecting assembly files...")

        asm_files = self.docker.find_files('*.s', out_dir, workdir=docker_work_dir)

        if asm_files:
            for idx, asm_file in enumerate(asm_files):
                content = self.docker.read_file(asm_file)
                if content:
                    asm_output = Path(test_log_dir) / f"{idx}.s"
                    with open(asm_output, 'w') as f:
                        f.write(content)
            if self.verbose:
                log_info(f"Copied {len(asm_files)} assembly file(s)")
        else:
            log_warning(f"No assembly files found in {out_dir}")

        # Copy before/after files
        if self.verbose:
            log_info("Collecting before/after files...")

        before_after_files = self.docker.list_files('*before.txt *after.txt', workdir=docker_work_dir)

        if before_after_files:
            for file in before_after_files:
                filename = Path(file).name
                content = self.docker.read_file(file)
                if content:
                    output_file = Path(test_log_dir) / filename
                    with open(output_file, 'w') as f:
                        f.write(content)
            if self.verbose:
                log_info(f"Copied {len(before_after_files)} before/after file(s)")

        # Parse results and save locally
        if self.verbose:
            log_info("Parsing results...")

        parse_cmd = [
            'python3', str(self.script_dir / 'parse_exe_time.py'),
            '--log-file', log_file,
            '--yaml-file', patched_yaml,
            '--output-dir', self.logs_dir,
            '--git-info', self.git_info,
            '--hostname', self.hostname,
            '--gpu-arch', self.gpu_arch,
            '--test-date', self.test_date,
            '--yaml-name', yaml_name
        ]
        if self.verbose:
            parse_cmd.append('--verbose')

        subprocess.run(parse_cmd, check=True)

        # Check if test passed
        test_passed = self.check_test_result(log_file, yaml_name)

        # Clean up docker workspace (unless --preserve-build is set)
        if not self.preserve_build:
            cleanup_pattern = f"{out_dir} *before.txt *after.txt a.out"
            self.docker.remove_files(cleanup_pattern, docker_work_dir)
        else:
            cleanup_pattern = "*before.txt *after.txt a.out"
            self.docker.remove_files(cleanup_pattern, docker_work_dir)
            if self.verbose:
                log_info(f"Preserved build directory: {out_dir}")

        return test_passed

    def execute_all(self) -> int:
        """Execute all execution time tests"""
        result = super().execute_all()

        # Update sia3cmp database if local-sia3cmp.json exists
        local_sia3cmp_json = Path(self.logs_dir) / "local-sia3cmp.json"
        if local_sia3cmp_json.exists():
            if self.verbose:
                log_info("Updating sia3cmp database...")

            sia3cmp_db_file = Path(self.database_dir) / f"sia3cmp-{self.hostname}_{self.gpu_arch}.json"

            cmd = [
                'python3', str(self.script_dir / 'update_sia3cmp_database.py'),
                '--local-json', str(local_sia3cmp_json),
                '--database-file', str(sia3cmp_db_file),
                '--git-hash', self.git_info
            ]
            if self.verbose:
                cmd.append('--verbose')

            subprocess.run(cmd, check=True)

            if self.verbose:
                from .utils import log_success
                log_success(f"Sia3cmp database updated: {sia3cmp_db_file}")

        return result

