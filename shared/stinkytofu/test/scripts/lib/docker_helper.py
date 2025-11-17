#!/usr/bin/env python3
"""
Docker helper functions for container operations
"""

import sys
import json
import subprocess
from typing import Optional, Dict, List, Tuple
from pathlib import Path
from .utils import log_info, log_error, log_success, log_warning


class DockerHelper:
    """Helper class for Docker operations"""

    def __init__(self, container_name: str, verbose: bool = False):
        self.container_name = container_name
        self.verbose = verbose

    def check_container(self) -> bool:
        """Check if container exists and is valid"""
        try:
            subprocess.run(
                ['docker', 'inspect', self.container_name],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            log_error(f"Container '{self.container_name}' does not exist")
            return False

    def start_container(self) -> bool:
        """Start container if not running"""
        try:
            subprocess.run(
                ['docker', 'start', self.container_name],
                capture_output=True,
                check=True
            )
            return True
        except subprocess.CalledProcessError:
            log_error(f"Failed to start container {self.container_name}")
            return False

    def detect_gpu_arch(self) -> Optional[str]:
        """Detect GPU architecture inside container"""
        try:
            result = subprocess.run(
                ['docker', 'exec', self.container_name,
                 '/opt/rocm/bin/rocm_agent_enumerator'],
                capture_output=True,
                text=True,
                check=True
            )
            lines = result.stdout.strip().split('\n')
            if len(lines) >= 2:
                gpu_arch = lines[1].strip()
                if gpu_arch and gpu_arch != "gfx000":
                    return gpu_arch

            log_error("Failed to detect valid GPU architecture")
            return None
        except Exception as e:
            log_error(f"Failed to detect GPU: {e}")
            return None

    def check_path_exists(self, path: str) -> bool:
        """Check if path exists inside container"""
        result = subprocess.run(
            ['docker', 'exec', self.container_name, 'test', '-e', path],
            capture_output=True
        )
        return result.returncode == 0

    def exec_command(self, cmd: str, workdir: Optional[str] = None,
                    interactive: bool = False, capture_output: bool = True) -> subprocess.CompletedProcess:
        """
        Execute command in Docker container

        Args:
            cmd: Command to execute (will be run in bash -c)
            workdir: Working directory inside container
            interactive: Whether to allocate a TTY
            capture_output: Whether to capture stdout/stderr

        Returns:
            CompletedProcess object
        """
        docker_cmd = ['docker', 'exec']

        if workdir:
            docker_cmd.extend(['-w', workdir])

        if interactive:
            docker_cmd.append('-it')

        docker_cmd.append(self.container_name)
        docker_cmd.extend(['bash', '-c', cmd])

        if self.verbose:
            log_info(f"Executing: {' '.join(docker_cmd)}")

        if capture_output:
            return subprocess.run(docker_cmd, capture_output=True, text=True)
        else:
            return subprocess.run(docker_cmd)

    def map_host_to_docker_path(self, host_path: str) -> Optional[str]:
        """
        Map host path to docker path using container mounts

        Args:
            host_path: Absolute path on host filesystem

        Returns:
            Corresponding path inside container, or None if no mapping found
        """
        try:
            # Get container mounts
            result = subprocess.run(
                ['docker', 'inspect', self.container_name,
                 '--format', '{{json .Mounts}}'],
                capture_output=True,
                text=True,
                check=True
            )
            mounts = json.loads(result.stdout)

            # Find best matching mount (longest common prefix)
            best_match_src = ""
            best_match_dst = ""
            best_match_len = 0

            for mount in mounts:
                source = mount.get('Source', '')
                destination = mount.get('Destination', '')

                if host_path.startswith(source):
                    if len(source) > best_match_len:
                        best_match_src = source
                        best_match_dst = destination
                        best_match_len = len(source)

            if best_match_src:
                relative_path = host_path[len(best_match_src):]
                return best_match_dst + relative_path

            log_error(f"Could not map host path '{host_path}' to docker path")
            return None

        except Exception as e:
            log_error(f"Failed to map path: {e}")
            return None

    def check_yappi(self) -> bool:
        """Check if yappi is installed in container"""
        result = self.exec_command("python3 -c 'import yappi; print(\"OK\")'")
        return result.returncode == 0

    def install_yappi(self) -> bool:
        """Install yappi in container"""
        if self.verbose:
            log_info("Installing yappi...")

        result = self.exec_command(
            "pip3 install --break-system-packages yappi",
            capture_output=(not self.verbose)
        )

        if result.returncode == 0:
            return True
        else:
            log_error("Failed to install yappi")
            if not self.verbose and result.stderr:
                print(result.stderr, file=sys.stderr)
            return False

    def find_files(self, pattern: str, search_dir: str, workdir: Optional[str] = None) -> List[str]:
        """Find files matching pattern in container directory"""
        cmd = f"find {search_dir} -type f -name '{pattern}' 2>/dev/null | sort || true"
        result = self.exec_command(cmd, workdir=workdir)

        if result.returncode == 0 and result.stdout:
            return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return []

    def list_files(self, pattern: str, workdir: Optional[str] = None) -> List[str]:
        """List files matching pattern in container directory (from workdir)"""
        cmd = f"ls -1 {pattern} 2>/dev/null | sort || true"
        result = self.exec_command(cmd, workdir=workdir)

        if result.returncode == 0 and result.stdout:
            return [f.strip() for f in result.stdout.strip().split('\n') if f.strip()]
        return []

    def read_file(self, file_path: str) -> Optional[str]:
        """Read file content from container"""
        result = self.exec_command(f"cat {file_path}")
        if result.returncode == 0:
            return result.stdout
        return None

    def remove_files(self, file_pattern: str, workdir: str) -> None:
        """Remove files matching pattern in container"""
        self.exec_command(f"rm -rf {file_pattern}", workdir=workdir)

