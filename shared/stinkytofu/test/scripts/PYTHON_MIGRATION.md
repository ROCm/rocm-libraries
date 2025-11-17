# Python Migration Guide

## Overview

The stinkytofu test infrastructure has been rewritten in Python for cross platform, better maintainability, error handling, and code organization.

## Changes

### Old Structure (Bash)
```
scripts/
├── main.sh                 # Main orchestrator
├── test_a_exe_time.sh     # Execution time tests
├── test_b_codegen_time.sh # Codegen time tests
├── test_c_dbg_verify.sh   # Debug verification tests
├── utils.sh               # Utility functions
└── example_run.sh         # Example runner
```

### New Structure (Python)
```
scripts/
├── test_runner.py         # Main orchestrator (replaces main.sh)
├── lib/
│   ├── __init__.py
│   ├── utils.py           # Core utilities
│   ├── docker_helper.py   # Docker operations
│   ├── build_system.py    # CMake build management
│   ├── test_executor.py   # Base test executor class
│   ├── test_list.py       # Test list parsing
│   ├── yaml_patcher.py    # YAML patching logic
│   ├── exe_time_test.py   # Execution time tests (replaces test_a)
│   ├── codegen_test.py    # Codegen tests (replaces test_b)
│   └── debug_test.py      # Debug tests (replaces test_c)
├── example_run.sh         # Entry point (now calls Python)
└── ... (other Python scripts unchanged)
```

## Benefits of Python Implementation

1. **Better Error Handling**: Structured exception handling instead of error-prone shell scripting
2. **Type Safety**: Optional type hints for better IDE support and documentation
3. **Code Reusability**: Object-oriented design with base classes and inheritance
4. **Easier Testing**: Can write unit tests for individual components
5. **Better Readability**: More structured code with clear class/function boundaries
6. **Improved Maintainability**: Easier to add features and fix bugs

## Usage

### Using example_run.sh (Recommended)

No changes needed! `example_run.sh` now calls `test_runner.py` internally:

```bash
./example_run.sh
```

### Direct Usage

You can also call `test_runner.py` directly:

```bash
python3 test_runner.py \
    --container-name my_container \
    --docker-path /workspace/hipblaslt \
    --host-path /home/user/hipblaslt \
    --output-dir ./results \
    --test all \
    --verbose
```

### Command Line Options

All options from the Bash version are supported:

- `--container-name`: Docker container name (required)
- `--docker-path`: Path to hipblaslt inside container (required)
- `--host-path`: Path to hipblaslt on host (required)
- `--output-dir`: Output directory for results (required)
- `--tool-path`: Path to stinkytofu test tools (optional)
- `--new-cmake`: Use new ROCm CMake system: 0=old, 1=new (default: 0)
- `--test`: Test type: exe_time, codegen_time, dbg_verify, or all (default: all)
- `--gpu-arch`: GPU architecture (auto-detected if not provided)
- `--pattern`: Regex pattern to filter tests (default: .*)
- `--preserve-build`: Keep build directories (default: clean up)
- `--verbose, -v`: Enable verbose output

## Backward Compatibility

The old Bash scripts are preserved but no longer used. If you need to revert:

```bash
git checkout HEAD~1 -- scripts/
```

## Migration Notes

### Key Differences

1. **No more eval**: Commands are constructed safely using subprocess
2. **Better path handling**: Uses Python's pathlib for robust path operations
3. **Cleaner Docker interaction**: DockerHelper class encapsulates all Docker operations
4. **Centralized configuration**: Config dictionary passed to all test executors
5. **Base class pattern**: Common logic in TestExecutor, specialized in subclasses

### Architecture

```
TestExecutor (Base Class)
├── ExeTimeTest    (exe_time tests)
├── CodegenTest    (codegen_time tests)
└── DebugTest      (dbg_verify tests)

Each test executor:
- Parses test lists
- Sets up test paths
- Executes tests
- Collects results
- Updates databases
- Prints summaries
```

### Docker Operations

All Docker operations go through `DockerHelper`:
- `exec_command()`: Execute commands in container
- `map_host_to_docker_path()`: Map host paths to container paths
- `check_path_exists()`: Check if path exists in container
- `find_files()`: Find files matching pattern
- `read_file()`: Read file from container

### Build System

`BuildSystem` class handles:
- CMake configuration (old and new systems)
- Build path determination
- Incremental vs full builds
- Client binary verification

## Testing

To verify the Python implementation works:

```bash
# Syntax check
python3 -m py_compile scripts/lib/*.py scripts/test_runner.py

# Show help
python3 scripts/test_runner.py --help

# Run a dry-run test (use a test container)
python3 scripts/test_runner.py \
    --container-name test_container \
    --docker-path /path/to/hipblaslt \
    --host-path /path/to/hipblaslt \
    --output-dir ./test_output \
    --test exe_time \
    --pattern "mfma.yaml" \
    --verbose
```

## Troubleshooting

### Import Errors

Make sure you're running from the correct directory:
```bash
cd /path/to/rocm-libraries_gfx12/shared/stinkytofu/test/scripts
python3 test_runner.py --help
```

### Path Issues

The Python code uses `realpath()` and `Path.resolve()` for robust path handling. Make sure all paths are accessible.

### Docker Errors

Check Docker is running and the container exists:
```bash
docker ps -a | grep your_container_name
```

## Future Enhancements

With the Python implementation, we can now easily add:

1. **Unit tests**: Test individual components in isolation
2. **Better logging**: Structured logging with log levels
3. **JSON/YAML config files**: Instead of shell variables
4. **Progress bars**: For long-running tests
5. **Parallel execution**: Run multiple tests concurrently
6. **Better reporting**: HTML reports, charts, graphs
7. **CI/CD integration**: Easier integration with Jenkins, GitLab CI, etc.

## Support

For issues or questions about the Python implementation, check:
1. This migration guide
2. Inline code documentation (docstrings)
3. QUICKSTART.md for general usage
4. Git history for the Bash implementation

