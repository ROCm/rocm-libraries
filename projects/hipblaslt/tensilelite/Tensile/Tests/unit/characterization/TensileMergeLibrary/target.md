# TensileMergeLibrary.py — characterization target (pure helpers)

Pins the pure logic-data helpers: ensurePath, allFiles, fixSizeInconsistencies,
addKernel (reuse/new), sanitizeSolutions, removeUnusedSolutions,
removeDuplicatedSolutions, msg/verbose/debug, findSolutionWithIndex,
compareDestFolderToYaml.

Pinned latent bug: fixSizeInconsistencies keys its dedup dict by
`(value for value in size)` — a *generator object*, unique per entry — so
duplicate sizes are NEVER merged (test asserts both kept).

Resistance (out of scope): reNameSolutions / compareProblemType (derive
ProblemType), mergeLogic, avoidRegressions, main — the CLI merge driver.
