# Repository Guidelines

## Project Structure & Module Organization
- `src/dnn_benchmarking/`: core library and CLI implementation.
- `src/dnn_benchmarking/cli/`: argument parsing and entry points.
- `src/dnn_benchmarking/execution/`: graph execution, timing, and buffer management.
- `src/dnn_benchmarking/graph/`: JSON graph loading and validation helpers.
- `src/dnn_benchmarking/validation/`: accuracy comparison and reference providers.
- `tests/`: unit and integration tests (`tests/unit`, `tests/integration`).
- `graphs/`: sample JSON graphs (for local benchmarking).

## Build, Test, and Development Commands
- Install (ROCm): `pip install -r requirements-rocm.txt -r requirements-dev.txt && pip install -e .`
- Install (CUDA): `pip install -r requirements-cuda.txt -r requirements-dev.txt && pip install -e .`
- Run CLI: `python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --warmup 10 --iters 100`
- A/B test: `python -m dnn_benchmarking --graph ./graphs/sample_conv_fwd.json --AId 1 --BId 2`
- Tests (CPU-only): `pytest -m "not gpu"`
- Tests (GPU): `LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest`
- Coverage: `LD_LIBRARY_PATH=/opt/rocm/lib:$LD_LIBRARY_PATH pytest --cov=dnn_benchmarking tests/`

## Coding Style & Naming Conventions
- Python project using `pyproject.toml`; target Python 3.9+.
- Follow PEP 8 style; 4-space indentation.
- Names: modules and functions use `snake_case`, classes use `PascalCase`.
- Prefer small, focused modules under the existing package layout.

## Testing Guidelines
- Framework: `pytest` with markers `gpu` and `slow`.
- Place unit tests under `tests/unit` and integration tests under `tests/integration`.
- Name tests as `test_*.py` and functions as `test_*`.
- Use `-m "not gpu"` for environments without hipDNN/ROCm.

## Commit & Pull Request Guidelines
- Commit messages in history are short, imperative, and capitalized (e.g., “Fix tests…”, “Add kernel timing…”).
- Keep commits focused; avoid mixing refactors with behavior changes.
- PRs should include a clear description, test commands run, and notes on GPU/driver requirements.
- If behavior changes affect CLI output, include a sample snippet in the PR description.

## Configuration & Environment Tips
- hipDNN Python bindings (`hipdnn_frontend`) are installed separately from your hipDNN build.
- GPU tests require ROCm libraries discoverable via `LD_LIBRARY_PATH=/opt/rocm/lib`.
