# Parser unit tests

Unit tests for the YAML-to-CMake parsers in `shared/ctest/`:

- `parse_test_categories.py` — GTest-pattern parser
- `parse_catch2_categories.py` — Catch2 tag-expression parser

## Layout

```
tests/
├── README.md                            # this file
├── conftest.py                          # sys.path wiring + shared CMake-output parser
├── fixtures/                            # representative sample test_categories.yaml files
│   ├── gtest_minimal.yaml               # smallest valid GTest YAML
│   ├── gtest_full.yaml                  # every feature: anchors, OS excludes, exclude_gpu, env
│   ├── gtest_no_gpu.yaml                # no exclude_gpu section
│   ├── gtest_empty_patterns.yaml        # category with empty test_patterns (warning case)
│   ├── gtest_invalid_pattern.yaml       # validator failure: unsafe gtest pattern
│   ├── gtest_invalid_identifier.yaml    # validator failure: unsafe label
│   ├── catch2_minimal.yaml              # smallest valid Catch2 YAML
│   ├── catch2_full.yaml                 # includes + excludes + [] sentinel + env
│   ├── catch2_invalid_tag.yaml          # validator failure: malformed tag
│   └── catch2_invalid_identifier.yaml   # validator failure: unsafe category name
├── test_parse_test_categories.py        # GTest parser tests
└── test_parse_catch2_categories.py      # Catch2 parser tests
```

## Running

Requires Python 3.8+ with `pytest` and `PyYAML`:

```bash
pip install pytest PyYAML
```

From the repository root (or anywhere; tests use absolute paths):

```bash
pytest -q shared/ctest/tests
```

To run a single test module:

```bash
pytest -q shared/ctest/tests/test_parse_test_categories.py
pytest -q shared/ctest/tests/test_parse_catch2_categories.py
```

To run a single test:

```bash
pytest -q shared/ctest/tests/test_parse_test_categories.py::test_cli_full_yaml_gpu_exclude_combines_with_category_exclude
```

## What is covered

Per parser, the suite covers:

1. **Validator helpers** — `validate_identifier`, `validate_gtest_pattern` /
   `validate_tag`, plus the full-config `validate_config` / `validate_categories`
   functions. Includes both happy-path acceptance and rejection of unsafe
   inputs, with parametrized cases for each character-class boundary.
2. **Internal logic** — `gpu_arch_matches` (hierarchical GFX matching), and
   `build_catch2_tag_expression` (the trickier of the two — Catch2's comma /
   space / `~` precedence means excludes must be duplicated across include
   clauses).
3. **YAML loader** — happy path, missing-file error path, malformed-YAML error
   path; both branches must exit non-zero.
4. **CLI behaviour** — every CLI invocation is via `subprocess.run` so we
   exercise the same path that `TestCategories.cmake` uses at configure time.
   We assert on:
   - Generated build-tree CMake stdout (parsed into a structured dict by
     `conftest.extract_add_test_blocks`).
   - Generated install-tree `CTestTestfile.cmake` (parsed by
     `conftest.parse_install_file`) — paths must be relative
     (`"../<target>"`) so the file works from `bin/<component>/` after
     install.
   - Exit codes for invalid input — validation failures must exit 1 and emit
     **no partial** CMake (atomicity).
   - Timeout multipliers, environment propagation, and OS-specific excludes.
   - GPU exclusion behaviour: hierarchical matching, per-(category, arch)
     suite emission, exclude composition (category excludes + GPU excludes).

The CLI tests deliberately go through `subprocess` rather than importing
`main()` because the parsers communicate solely through argv / stdout / stderr /
exit code and that contract is what `TestCategories.cmake` depends on.
