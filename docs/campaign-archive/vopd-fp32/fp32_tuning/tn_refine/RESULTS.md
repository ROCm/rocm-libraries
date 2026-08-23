# TN Refinement Results — WGM / DepthU / TransposeLDS

Surviving DepthU values (answers DU=24 validity): [8, 16, 24, 32]
Surviving WGM values: [1, 4, 8, 16, 32]
Surviving TransposeLDS values: [0]

| shape | best GF (cache-warm) | MT | DU | WGM | TLDS |
|---|---:|---|---:|---:|---:|
| 4096x4096x4096 | 23765 | MT128x256x16 | 16 | 4 | 0 |
| 8192x8192x8192 | 32249 | MT128x256x16 | 16 | 1 | 0 |
| 2048x8192x8192 | 28108 | MT128x256x16 | 16 | 16 | 0 |
| 1024x8192x8192 | 22958 | MT128x256x8 | 8 | 1 | 0 |
| 1024x14336x4096 | 21956 | MT128x256x16 | 16 | 4 | 0 |
| 1024x4096x14336 | 21502 | MT128x256x8 | 8 | 1 | 0 |
| 4096x8192x8192 | 30745 | MT128x256x8 | 8 | 1 | 0 |
| 2048x4096x8192 | 23027 | MT128x256x8 | 8 | 1 | 0 |
| 3072x4096x4096 | 22340 | MT256x128x8 | 8 | 4 | 0 |
| 6144x4096x4096 | 27803 | MT128x256x16 | 16 | 4 | 0 |
| 2048x2048x8192 | 18427 | MT128x256x16 | 16 | 4 | 0 |
| 512x4096x4096 | 11492 | MT128x128x8 | 8 | 1 | 0 |

## Winning-lever summary (across these shapes)
- DU among winners: {16: 6, 8: 6}
- WGM among winners: {4: 5, 1: 6, 16: 1}
- TransposeLDS among winners: {0: 12}

NOTE: cache-warm Tensile numbers. Confirm any new lever (WGM not in {4,8}, DU=24/48,
or TLDS=1) with cold-cache hipblaslt-bench A/B vs the deployed TN kernel before adopting.