# LibraryLogic.py — characterization target (pure helpers)

LibraryLogic.py is dominated by the LogicAnalyzer class (~1200 lines of
benchmark-data analysis over per-problem winner arrays) plus analyzeProblemType
and generateLogic, all of which require fully-derived Solution/ProblemType state
and parsed CSV benchmark data (codegen/analysis), out of scope here.

This suite pins the two pure helpers:
- handle_frequency_issue (interactive input loop: empty / non-positive /
  invalid / valid)
- read_max_freq (MAX_FREQ env: set / unset / empty / invalid)

Resistance (integration-test-covered only): LogicAnalyzer (all methods),
analyzeProblemType, generateLogic, main.
