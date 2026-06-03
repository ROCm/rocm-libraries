# Characterization target — `Tensile/Common/Architectures.py`

Part of the master-plan remaining-module sweep. Adds direct coverage of the pure
helpers (supportsChipIdPredicate, isaToGfx/gfxToIsa incl. hex step + invalid,
gfxToSwCodename in-map/substring/None, gfxToVariants fallback, cliArchsToIsa
;/_/all) and `_detectGlobalCurrentISA` (subprocess `run` monkeypatched, success
+ failure). The file-parsing / predicate-filtering functions (_extractArchInfo,
splitArchsFromPredicates, variant maps, filterLogicFilesByPredicates) are
already exercised by the existing suite; combined ≥95% verified at the Batch C
checkpoint. 10 tests.
