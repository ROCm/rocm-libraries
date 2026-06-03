# SolutionLibrary.py — characterization target

Pins the library (de)serialization class tree: SingleSolutionLibrary,
IndexSolutionLibrary, PlaceholderLibrary (+merge collision), MatchingLibrary
(all distance variants), FreeSize/Prediction/MLPClassification (+MLP merge
raises), ProblemMapLibrary, PredicateLibrary (both sort branches), and
MasterSolutionLibrary: hardware() (fallback + real-arch + gfx950 chip-id
suffix), FixSolutionIndices, state/cpp/remap/merge (incl. lazyLibraries),
applyNaming, BenchmarkingLibrary, and FromOriginalState (non-lazy + lazy
placeholder recursion with all placeholder-suffix toggles).

Coverage: 413 stmts, 12 missed → 97.1% line (96.12% blended).

Heavy integration is driven with fakes: Contractions.ProblemType/Solution
derivation is stubbed, solutionClass is a fake whose FromSolutionStruct yields
index-bearing fakes, and a rich fake ProblemType exercises the lazy
placeholder-name suffix block.

Residual misses (12 lines): merge-assert alternates (134/179/215/250), the
PerfMetric != DeviceEfficiency branch (432), predicates lazy alt (439, 463-466),
chip-id non-list branch (377), useScaleAB=="Vector" (502), and useE/_Aux
(non-gradient) branch (528) — each an alternate arm whose primary arm is pinned.
