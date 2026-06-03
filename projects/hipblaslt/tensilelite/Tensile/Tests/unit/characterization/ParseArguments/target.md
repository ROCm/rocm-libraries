# TensileCreateLibrary/ParseArguments.py — characterization target

Pins the argv -> arguments-dict mapping of `parseArguments`: defaults, every
store_true/store_false flag, value-carrying options, code-object-version
mapping, the CMAKE_CXX_COMPILER env side effect, and argparse validation exits.

Coverage: 65 stmts, 0 missed → 100% line.

Pinned quirk: `parseArguments(input)` ignores its `input` parameter and parses
`sys.argv` directly (argparse `.parse_args()` with no list).

Coverage note: `--cov=Tensile.TensileCreateLibrary.ParseArguments` (dotted)
aborts (rocisa nanobind double-import SIGABRT); use the package-path form
`--cov=Tensile` and grep the row instead.
