# Parser unit tests

Python `unittest` suites for the YAML-to-CMake parsers in `shared/ctest/`:

- `parse_test_categories.py` — GTest-filter parser → `test_parse_test_categories.py`
- `parse_catch2_categories.py` — Catch2 tag-expression parser → `test_parse_catch2_categories.py`
- `parse_ctest_categories.py` — pre-registered-CTest label parser → `test_parse_ctest_categories.py`

## Running

The tests use only the Python standard library (`unittest`) plus `PyYAML`
(already required by the parsers). No pytest dependency.

Run a single suite directly:

```bash
cd shared/ctest
python3 -m unittest discover -v -s tests -p "test_parse_catch2_categories.py"
```

Run every parser suite:

```bash
cd shared/ctest
python3 -m unittest discover -v -s tests -p "test_parse_*.py"
```

Via CTest (each suite is registered as a separate test in `CMakeLists.txt`):

```bash
ctest -R shared_ctest_parse_.*_unit
```

## Conventions

Each test module is self-contained (mirroring `test_parse_test_categories.py`):

- Inserts `shared/ctest` onto `sys.path` and imports the parser module for
  direct unit tests of helper/validator functions.
- Defines YAML inputs inline as string constants (no external fixture files),
  writing them to a `tempfile.TemporaryDirectory()` for CLI tests.
- Runs the parser as a subprocess (`sys.executable <parser> ...`) so the tests
  exercise the exact argv/stdout/exit-code contract that `TestCategories.cmake`
  depends on, and inspects the generated build-tree CMake (stdout) and
  install-tree `CTestTestfile.cmake` (relative-path form).

When adding a new parser, add a matching `test_parse_<name>.py` and a
corresponding `add_test(... -p "test_parse_<name>.py")` block in
`CMakeLists.txt`.
