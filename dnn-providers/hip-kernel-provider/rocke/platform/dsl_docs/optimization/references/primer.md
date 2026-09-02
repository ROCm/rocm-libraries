# Primer

Start here when the question is broad. All ids resolve with `get_page.py`.

## Route

1. `process-routing` then `family-overview`
2. Operator table: `family-gemm` / `family-attention` / `family-convolution` / `family-moe` / `family-small-ops`
3. Symptom: `queries/by-problem.md`
4. gfx: `hw-gfx942` `hw-gfx950` `hw-gfx1151` `hw-gfx1201` `hw-gfx1250`
5. Catalog stall: `process-escape-hatch` (`--symptom catalog-exhausted`)
6. gfx1250 / Blackwell-shaped: `hw-gfx1250` then `queries/by-migration.md`

## Common techniques

`technique-tiling`, `technique-software-pipeline`, `technique-lds-swizzle`,
`technique-async-copy`, `technique-vectorized-io`, `technique-occupancy`,
`technique-epilogue`, `technique-persistent-streamk`, `technique-fusion`,
`technique-isa-inspect`, `technique-algorithm-break` (escape hatch only)

## Arch-specific

- CDNA: `technique-mfma-atom`, `technique-ds-read-tr`, `technique-agpr-acc`, `technique-chiplet-swizzle`
- RDNA: `technique-wmma-atom`, `technique-wave32`
- gfx1250: `technique-gfx1250-wmma-k32`, `technique-gfx12-async-lds`, `technique-gfx1250-ds-load-tr`, `technique-gfx1250-asynccnt-pipeline`, `technique-gfx1250-split-waitcnt`

## Sources in rocm-libraries

`project-hipblaslt`, `project-tensile`, `project-tensilelite`,
`project-stinkytofu`, `project-rocroller`, `project-origami`, `project-miopen`,
`project-hipdnn`, `project-composablekernel`, `project-rocke`
