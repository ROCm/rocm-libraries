#!/usr/bin/env python3
# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""TheRock developer bootstrap and build tool.

Downloads prebuilt CI artifacts and lets you selectively build
ml-libs components (hipdnn, hipkernelprovider, etc.) from source
while everything else uses prebuilt binaries.

Cross-platform replacement for rock_dev_bootstrap.sh.

Usage:
    python rock_dev_bootstrap.py <command> [options] [components...]

Commands:
    bootstrap [run-id]              Download CI artifacts (auto-detects latest if omitted)
    configure [components...]       Remove .prebuilt markers + run cmake
    build [components...]           Build components with ninja
    rebuild [components...]         Expunge + rebuild components

Examples:
    # One-time setup (auto-detect latest nightly):
    python rock_dev_bootstrap.py bootstrap --therock-dir /path/to/TheRock

    # One-time setup with specific run:
    python rock_dev_bootstrap.py bootstrap --therock-dir /path/to/TheRock 22884930750

    # Configure to build hipdnn + miopenprovider from source:
    python rock_dev_bootstrap.py configure hipdnn miopenprovider

    # Build everything that was configured:
    python rock_dev_bootstrap.py build

    # Clean rebuild of a component:
    python rock_dev_bootstrap.py rebuild hipkernelprovider

Available components:
    hipdnn, hipkernelprovider, miopenprovider, hipblasltprovider
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ALL_COMPONENTS = [
    "hipdnn",
    "hipkernelprovider",
    "miopenprovider",
    "hipblasltprovider",
    "miopen",
]

NINJA_TARGETS = {
    "hipdnn": "hipDNN",
    "hipkernelprovider": "hipkernelprovider",
    "miopenprovider": "miopenprovider",
    "hipblasltprovider": "hipblasltprovider",
    "miopen": "MIOpen",
}

# Platform-aware defaults
if sys.platform == "win32":
    _DEFAULT_GPU = "gfx1151"
    _DEFAULT_PLATFORM = "windows"
else:
    _DEFAULT_GPU = "gfx94X-dcgpu"
    _DEFAULT_PLATFORM = "linux"

# ci_nightly.yml is the top-level workflow that triggers both Linux and Windows
# builds.  Reusable workflows (ci_windows.yml, ci_linux.yml) don't have their
# own runs in the GitHub API, so we must search the nightly workflow.
_DEFAULT_WORKFLOW = "ci_nightly.yml"

CONFIG_FILENAME = ".dev_bootstrap_config.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def component_to_flag(name: str) -> str:
    """Convert component name to cmake enable flag (e.g. hipdnn -> HIPDNN)."""
    return name.upper().replace("-", "_")


def component_to_ninja_target(name: str) -> str:
    """Convert component name to ninja target name."""
    return NINJA_TARGETS.get(name, name)


def validate_component(name: str) -> None:
    """Exit with error if component name is not recognized."""
    if name not in ALL_COMPONENTS:
        print(f"ERROR: Unknown component '{name}'")
        print(f"Available components: {', '.join(ALL_COMPONENTS)}")
        sys.exit(1)


def gpu_short(gpu_family: str) -> str:
    """Extract short GPU name (strip everything after first hyphen)."""
    return gpu_family.split("-")[0]


def config_path(build_dir: Path) -> Path:
    """Path to the saved config file in the build directory."""
    return build_dir / CONFIG_FILENAME


def save_config(
    build_dir: Path, gpu: str, components: list[str], therock_dir: Path = None
) -> None:
    """Save current settings to the build directory for later commands."""
    build_dir.mkdir(parents=True, exist_ok=True)
    data = {"gpu": gpu, "components": components}
    if therock_dir is not None:
        data["therock_dir"] = str(therock_dir)
    config_path(build_dir).write_text(json.dumps(data, indent=2))


def load_config(build_dir: Path) -> dict:
    """Load saved config from the build directory."""
    cfg = config_path(build_dir)
    if cfg.exists():
        return json.loads(cfg.read_text())
    return {}


def venv_python(therock_dir: Path) -> Path:
    """Return path to the venv's Python interpreter."""
    venv_dir = therock_dir / ".venv"
    if sys.platform == "win32":
        return venv_dir / "Scripts" / "python.exe"
    return venv_dir / "bin" / "python"


def run_cmd(
    cmd: list, cwd: Path = None, check: bool = True, env: dict = None
) -> subprocess.CompletedProcess:
    """Run a command, printing it first. Exits on failure if check=True."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, env=env)
    if check and result.returncode != 0:
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


def run_cmd_capture(cmd: list, cwd: Path = None) -> subprocess.CompletedProcess:
    """Run a command and capture stdout. Exits on failure."""
    print(f"  $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        print(f"ERROR: Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result


# ---------------------------------------------------------------------------
# MSVC environment
# ---------------------------------------------------------------------------

_VCVARS_SEARCH_ROOTS = [
    Path("D:/develop"),
    Path("C:/Program Files/Microsoft Visual Studio"),
    Path("C:/Program Files (x86)/Microsoft Visual Studio"),
]


def find_vcvars64() -> Path:
    """Locate vcvars64.bat by searching known root directories."""
    for root in _VCVARS_SEARCH_ROOTS:
        if not root.is_dir():
            continue
        for bat in root.rglob("vcvars64.bat"):
            return bat
    return None


def get_msvc_env() -> dict:
    """Capture MSVC environment by running vcvars64.bat and reading env vars.

    Returns an env dict with MSVC variables merged into the current environment,
    or None on non-Windows or if vcvars64.bat cannot be found.
    """
    if sys.platform != "win32":
        return None

    # If cl.exe is already available, no need to activate
    if shutil.which("cl"):
        return None

    vcvars = find_vcvars64()
    if vcvars is None:
        print("WARNING: Cannot find vcvars64.bat — MSVC may not be available.")
        print("  Run this script from a 'x64 Native Tools Command Prompt for")
        print("  VS 2022', or install Visual Studio Build Tools.")
        return None

    print(f"Activating MSVC environment via {vcvars}...")
    # Run vcvars64.bat in a cmd shell, then dump the resulting environment.
    # Use a .bat wrapper written to a temp file to avoid quoting issues.
    import tempfile

    bat_content = f'@call "{vcvars}" >nul 2>&1\n@if errorlevel 1 exit /b 1\n@set\n'
    bat_file = Path(tempfile.gettempdir()) / "_rock_dev_vcvars.bat"
    bat_file.write_text(bat_content)
    try:
        result = subprocess.run(
            ["cmd.exe", "/c", str(bat_file)], capture_output=True, text=True
        )
    finally:
        bat_file.unlink(missing_ok=True)

    if result.returncode != 0:
        print(f"WARNING: vcvars64.bat failed (exit {result.returncode})")
        if result.stderr:
            print(f"  {result.stderr.strip()}")
        return None

    env = {}
    for line in result.stdout.splitlines():
        if "=" in line:
            key, _, value = line.partition("=")
            env[key] = value

    # Ensure "." is on PATH so cmd.exe can find batch files in the current
    # working directory.  This is default cmd.exe behavior but breaks when
    # processes are spawned from Git Bash.
    path = env.get("Path", env.get("PATH", ""))
    if not path.startswith(".;"):
        env["Path"] = ".;" + path

    return env


def get_build_env(therock_dir: Path) -> dict:
    """Build an environment dict for cmake/ninja with MSVC and venv Python.

    Ensures the TheRock venv's Python (which has pyyaml and other deps from
    requirements.txt) is on PATH so cmake's find_package(Python3) finds it
    instead of the system Python.
    """
    env = get_msvc_env()
    if env is None:
        env = dict(os.environ)

    venv_dir = therock_dir / ".venv"
    if sys.platform == "win32":
        venv_bin = str(venv_dir / "Scripts")
    else:
        venv_bin = str(venv_dir / "bin")

    path = env.get("Path", env.get("PATH", ""))
    if venv_bin not in path:
        env["Path"] = venv_bin + ";" + path

    return env


# ---------------------------------------------------------------------------
# Python venv setup
# ---------------------------------------------------------------------------


def setup_python(therock_dir: Path) -> Path:
    """Ensure a Python venv exists in TheRock dir with deps installed.

    Returns the path to the venv's Python interpreter.
    """
    venv_dir = therock_dir / ".venv"
    python = venv_python(therock_dir)
    requirements = therock_dir / "requirements.txt"

    if not venv_dir.exists():
        print("Creating Python virtual environment...")
        run_cmd([sys.executable, "-m", "venv", str(venv_dir)])

    print("Installing dependencies...")
    run_cmd([str(python), "-m", "pip", "install", "-q", "--upgrade", "pip"])
    if requirements.exists():
        run_cmd([str(python), "-m", "pip", "install", "-q", "-r", str(requirements)])

    return python


# ---------------------------------------------------------------------------
# Prebuilt marker removal
# ---------------------------------------------------------------------------


def remove_prebuilt_markers(build_dir: Path, components: list[str]) -> None:
    """Remove .prebuilt marker files for the given components."""
    print()
    print("Removing .prebuilt markers for source-build components...")
    print()

    for comp in components:
        found = False
        for prebuilt in build_dir.rglob("stage.prebuilt"):
            if comp.lower() in str(prebuilt).lower():
                found = True
                print(f"  {comp}: removing .prebuilt marker ({prebuilt})")
                prebuilt.unlink()
                stage_dir = prebuilt.with_suffix("")
                if stage_dir.is_dir():
                    shutil.rmtree(stage_dir)
        if not found:
            print(f"  {comp}: no .prebuilt marker (will build from source)")
    print()


# ---------------------------------------------------------------------------
# Commands
# ---------------------------------------------------------------------------


def do_bootstrap(args: argparse.Namespace) -> None:
    """Download CI artifacts and set up prebuilt markers."""
    therock_dir = args.therock_dir
    build_dir = args.build_dir
    gpu = args.gpu
    workflow = args.workflow
    platform = args.platform
    run_id = args.run_id

    # Validate TheRock dir
    artifact_manager = therock_dir / "build_tools" / "artifact_manager.py"
    if not artifact_manager.exists():
        print(f"ERROR: Cannot find {artifact_manager}")
        print("Make sure --therock-dir points to a TheRock checkout.")
        sys.exit(1)

    python = setup_python(therock_dir)

    # Always fetch sources to ensure they're up to date
    print("Fetching submodule sources...")
    fetch_cmd = [str(python), str(therock_dir / "build_tools" / "fetch_sources.py")]
    if shutil.which("dvc") is None:
        print()
        print("  WARNING: 'dvc' not found on PATH — skipping large file pulls.")
        print("  Some components may fail to build if they require DVC-managed files.")
        print("  Install DVC from https://dvc.org/doc/install if needed.")
        print()
        fetch_cmd.extend(["--dvc-projects", "_skip"])
    run_cmd(fetch_cmd, cwd=therock_dir)
    print()

    # Find latest run ID if not specified
    if not run_id:
        print("==========================================")
        print("Finding latest CI run with artifacts")
        print("==========================================")
        print()
        print(f"Searching {workflow} for {gpu} artifacts...")
        print()

        result = run_cmd_capture(
            [
                str(python),
                str(therock_dir / "build_tools" / "find_latest_artifacts.py"),
                "--artifact-group",
                gpu,
                "--workflow",
                workflow,
                "--platform",
                platform,
                "--verbose",
            ],
            cwd=therock_dir,
        )

        output = result.stdout + result.stderr
        print(output)

        match = re.search(r"Workflow run ID:\s+(\d+)", output)
        if not match:
            print()
            print(f"ERROR: Could not find a recent CI run with artifacts for {gpu}")
            print()
            print("Try specifying a run ID manually:")
            print(f"  python {sys.argv[0]} bootstrap <run-id>")
            sys.exit(1)

        run_id = match.group(1)
        print(f"Using run ID: {run_id}")

    print()
    print("==========================================")
    print(f"Fetching all artifacts from run {run_id}")
    print("==========================================")
    print()
    print(f"GPU Family: {gpu}")
    print(f"Build Dir:  {build_dir}")
    print()

    run_cmd(
        [
            str(python),
            str(therock_dir / "build_tools" / "artifact_manager.py"),
            "fetch",
            "--stage",
            "all",
            "--run-id",
            run_id,
            "--amdgpu-families",
            gpu,
            "--output-dir",
            str(build_dir),
            "--bootstrap",
            "--platform",
            platform,
        ],
        cwd=therock_dir,
    )

    prebuilt_count = len(list(build_dir.rglob("*.prebuilt")))
    print()
    print("==========================================")
    print("Bootstrap complete!")
    print("==========================================")
    print()
    print(f"  {prebuilt_count} components marked as prebuilt")
    print()
    print(f"Next: python {sys.argv[0]} configure [components...]")
    print()


def do_configure(args: argparse.Namespace) -> None:
    """Remove .prebuilt markers and run cmake configure."""
    therock_dir = args.therock_dir
    build_dir = args.build_dir
    gpu = args.gpu
    components = args.components if args.components else list(ALL_COMPONENTS)

    python = setup_python(therock_dir)
    build_env = get_build_env(therock_dir)

    print("==========================================")
    print("Configuring TheRock build")
    print("==========================================")
    print()
    print(f"GPU Family: {gpu}")
    print(f"Build Dir:  {build_dir}")
    print()
    print("Building from source:")
    for comp in components:
        print(f"  - {comp}")

    # Remove .prebuilt markers
    remove_prebuilt_markers(build_dir, components)

    # Save config
    save_config(build_dir, gpu, components, therock_dir)

    # Build cmake args
    cmake_args = [
        "cmake",
        "-B",
        str(build_dir),
        "-GNinja",
        "-DTHEROCK_ENABLE_ALL=OFF",
        f"-DTHEROCK_AMDGPU_FAMILIES={gpu}",
        "-DCMAKE_BUILD_TYPE=RelWithDebInfo",
    ]
    for comp in components:
        flag = component_to_flag(comp)
        cmake_args.append(f"-DTHEROCK_ENABLE_{flag}=ON")

    print("Running cmake...")
    run_cmd(cmake_args, cwd=therock_dir, env=build_env)

    print()
    print("==========================================")
    print("Configure complete!")
    print("==========================================")
    print()
    print(f"Next: python {sys.argv[0]} build [components...]")
    print()


def do_build(args: argparse.Namespace) -> None:
    """Build configured components with ninja."""
    build_dir = args.build_dir
    components = args.components

    # Load saved config if no components specified
    cfg = load_config(build_dir)
    if not components:
        components = cfg.get("components", [])

    therock_dir = args.therock_dir or (
        Path(cfg["therock_dir"]) if "therock_dir" in cfg else None
    )
    build_env = get_build_env(therock_dir) if therock_dir else get_msvc_env()

    if not components:
        print("ERROR: No components specified and no saved config found.")
        print(f"Run 'python {sys.argv[0]} configure [components...]' first.")
        sys.exit(1)

    targets = [component_to_ninja_target(c) for c in components]

    print("==========================================")
    print("Building components")
    print("==========================================")
    print()
    print(f"Build Dir: {build_dir}")
    print(f"Targets:   {' '.join(targets)}")
    print()

    ninja_cmd = ["ninja", "-C", str(build_dir)]
    if args.jobs:
        ninja_cmd.extend(["-j", str(args.jobs)])
    run_cmd(ninja_cmd + targets, env=build_env)

    print()
    print("Build complete!")
    print()


def do_rebuild(args: argparse.Namespace) -> None:
    """Expunge and rebuild components."""
    build_dir = args.build_dir
    components = args.components

    # Load saved config if no components specified
    cfg = load_config(build_dir)
    if not components:
        components = cfg.get("components", [])

    therock_dir = args.therock_dir or (
        Path(cfg["therock_dir"]) if "therock_dir" in cfg else None
    )
    build_env = get_build_env(therock_dir) if therock_dir else get_msvc_env()

    if not components:
        print("ERROR: No components specified and no saved config found.")
        print(f"Run 'python {sys.argv[0]} configure [components...]' first.")
        sys.exit(1)

    targets = [component_to_ninja_target(c) for c in components]
    expunge_targets = [f"{t}+expunge" for t in targets]

    print("==========================================")
    print("Rebuilding components (expunge + build)")
    print("==========================================")
    print()
    print(f"Build Dir: {build_dir}")
    print(f"Targets:   {' '.join(targets)}")
    print()

    ninja_cmd = ["ninja", "-C", str(build_dir)]
    if args.jobs:
        ninja_cmd.extend(["-j", str(args.jobs)])

    print(f"Expunging: {' '.join(expunge_targets)}")
    run_cmd(ninja_cmd + expunge_targets, env=build_env)

    print()
    print(f"Building: {' '.join(targets)}")
    run_cmd(ninja_cmd + targets, env=build_env)

    print()
    print("Rebuild complete!")
    print()


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def parse_args(argv: list[str] = None) -> argparse.Namespace:
    """Parse command line arguments."""
    # Shared parent parser so global options work before or after the command
    global_opts = argparse.ArgumentParser(add_help=False)
    global_opts.add_argument(
        "--gpu",
        default=_DEFAULT_GPU,
        help=f"GPU family (default: {_DEFAULT_GPU})",
    )
    global_opts.add_argument(
        "--build-dir",
        type=Path,
        default=None,
        help="Build directory (default: <cwd>/build/therock-<gpu-short>)",
    )
    global_opts.add_argument(
        "--therock-dir",
        type=Path,
        default=None,
        help="Path to TheRock checkout (required for bootstrap/configure)",
    )
    global_opts.add_argument(
        "--workflow",
        default=_DEFAULT_WORKFLOW,
        help=f"CI workflow file for artifact lookup (default: {_DEFAULT_WORKFLOW})",
    )
    global_opts.add_argument(
        "--platform",
        default=_DEFAULT_PLATFORM,
        help=f"Target platform (default: {_DEFAULT_PLATFORM})",
    )
    global_opts.add_argument(
        "-j",
        "--jobs",
        type=int,
        default=None,
        help="Number of parallel build jobs (passed to ninja -j)",
    )

    parser = argparse.ArgumentParser(
        prog="rock_dev_bootstrap.py",
        description="TheRock developer bootstrap and build tool.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        parents=[global_opts],
        epilog="""
Available components:
  hipdnn, hipkernelprovider, miopenprovider, hipblasltprovider

Per-component ninja targets (for manual use):
  component+configure  - Run cmake configure
  component+build      - Incremental build
  component+stage      - Install to stage dir
  component+expunge    - Full clean (build/, stage/, stamp/, dist/)
""",
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # bootstrap
    bp = subparsers.add_parser(
        "bootstrap",
        parents=[global_opts],
        help="Download CI artifacts (auto-detects latest if run-id omitted)",
    )
    bp.add_argument(
        "run_id",
        nargs="?",
        default=None,
        help="GitHub Actions run ID (auto-detected if omitted)",
    )

    # configure
    cp = subparsers.add_parser(
        "configure",
        parents=[global_opts],
        help="Remove .prebuilt markers + run cmake",
    )
    cp.add_argument(
        "components",
        nargs="*",
        default=[],
        help="Components to build from source (default: all)",
    )

    # build
    bp2 = subparsers.add_parser(
        "build",
        parents=[global_opts],
        help="Build components with ninja",
    )
    bp2.add_argument(
        "components",
        nargs="*",
        default=[],
        help="Components to build (default: from saved config)",
    )

    # rebuild
    rp = subparsers.add_parser(
        "rebuild",
        parents=[global_opts],
        help="Expunge + rebuild components",
    )
    rp.add_argument(
        "components",
        nargs="*",
        default=[],
        help="Components to rebuild (default: from saved config)",
    )

    args = parser.parse_args(argv)

    if not args.command:
        parser.print_help()
        sys.exit(0)

    # Derive build dir from GPU if not set.
    # Default: <cwd>/build/therock-<gpu> — builds land relative to where
    # the script is invoked.  Use --build-dir to override.
    if args.build_dir is None:
        args.build_dir = Path.cwd() / "build" / f"therock-{gpu_short(args.gpu)}"

    # Validate components
    components = getattr(args, "components", None)
    if components:
        for comp in components:
            validate_component(comp)

    # Require therock-dir for commands that need it
    if args.command in ("bootstrap", "configure"):
        if args.therock_dir is None:
            print(
                "ERROR: --therock-dir is required for the '{}' command.".format(
                    args.command
                )
            )
            sys.exit(1)
        args.therock_dir = args.therock_dir.resolve()
        if not args.therock_dir.is_dir():
            print(f"ERROR: TheRock directory does not exist: {args.therock_dir}")
            sys.exit(1)

    return args


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = parse_args()

    dispatch = {
        "bootstrap": do_bootstrap,
        "configure": do_configure,
        "build": do_build,
        "rebuild": do_rebuild,
    }

    handler = dispatch.get(args.command)
    if handler:
        handler(args)
    else:
        print(f"ERROR: Unknown command '{args.command}'")
        sys.exit(1)


if __name__ == "__main__":
    main()
