# OOB representatives

2–3 sizes per `OOB category` from `hipblaslt-ZeroGPUDay workloads(BBS TN OOB exact re-tune_1027C).csv`
(low / mid / high flops, unique M×N×K×batch). Used by `bbs_tn_oob_param_groups.yaml`.

| Category | index | M | N | K | batch | macro_tile |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| 01. Small GEMMs | 44 | 656 | 112 | 32 | 1 | 64x16x32_0 |
| 01. Small GEMMs | 69 | 96 | 224 | 240 | 1 | 32x32x64_0 |
| 01. Small GEMMs | 194 | 256 | 224 | 144 | 1 | 32x32x64_0 |
| 02. Medium GEMMs | 129 | 464 | 1872 | 6736 | 1 | 160x192x64_0 |
| 02. Medium GEMMs | 56 | 544 | 1728 | 7488 | 1 | 192x176x64_0 |
| 02. Medium GEMMs | 191 | 7056 | 896 | 1248 | 1 | 208x128x128_0 |
| 03. Large GEMMs | 102 | 4464 | 5968 | 6464 | 1 | 224x256x128_0 |
| 03. Large GEMMs | 166 | 4928 | 4848 | 7216 | 1 | 192x256x128_0 |
| 03. Large GEMMs | 105 | 5504 | 6096 | 5440 | 1 | 176x256x128_0 |
| 04. Large M, smaller N and K | 193 | 11600 | 7936 | 6048 | 1 | 240x256x128_0 |
| 04. Large M, smaller N and K | 53 | 315952 | 3696 | 1424 | 1 | 256x256x128_0 |
| 04. Large M, smaller N and K | 55 | 168464 | 3200 | 6368 | 1 | 256x256x128_0 |
| 05. Large M, very small N and K | 177 | 259280 | 16 | 32 | 1 | 256x16x32_0 |
| 05. Large M, very small N and K | 104 | 98000 | 96 | 64 | 1 | 192x96x32_0 |
| 05. Large M, very small N and K | 38 | 142816 | 80 | 64 | 1 | 192x80x32_0 |
| 06. Large N, smaller M and K | 97 | 1696 | 107104 | 928 | 1 | 256x256x128_0 |
| 06. Large N, smaller M and K | 150 | 4944 | 239104 | 192 | 1 | 224x256x64_0 |
| 06. Large N, smaller M and K | 120 | 688 | 957600 | 384 | 1 | 256x224x64_0 |
| 07. Large N, very small M and K | 17 | 16 | 450560 | 32 | 1 | 16x256x32_0 |
| 07. Large N, very small M and K | 172 | 32 | 207648 | 64 | 1 | 32x224x32_0 |
| 07. Large N, very small M and K | 27 | 16 | 434352 | 64 | 1 | 16x256x32_0 |
| 08. Large K, smaller M and N | 45 | 1472 | 3232 | 10448 | 1 | 192x208x64_0 |
| 08. Large K, smaller M and N | 157 | 1392 | 1680 | 319312 | 1 | 208x256x128_0 |
| 08. Large K, smaller M and N | 109 | 7104 | 1648 | 80048 | 1 | 224x224x128_0 |
| 09. Large K, very small M and N | 10 | 16 | 64 | 231696 | 1 | 16x64x128_0 |
| 09. Large K, very small M and N | 36 | 80 | 80 | 49872 | 1 | 80x128x64_0 |
| 09. Large K, very small M and N | 110 | 32 | 32 | 423200 | 1 | 32x32x256_0 |
| 10. Large M and N | 107 | 24208 | 32416 | 2336 | 1 | 256x256x128_0 |
| 10. Large M and N | 98 | 26656 | 47824 | 1728 | 1 | 256x256x128_0 |
| 10. Large M and N | 190 | 25520 | 15760 | 7296 | 1 | 256x256x128_0 |
| 11. Large N and K | 30 | 368 | 8432 | 30048 | 1 | 192x208x128_0 |
| 11. Large N and K | 153 | 1040 | 10928 | 28400 | 1 | 224x224x128_0 |
| 11. Large N and K | 61 | 912 | 13120 | 53472 | 1 | 256x208x128_0 |
| 12. Large M and K | 68 | 35216 | 208 | 19952 | 1 | 144x256x128_0 |
| 12. Large M and K | 67 | 51904 | 848 | 31376 | 1 | 256x176x128_0 |
| 12. Large M and K | 19 | 25680 | 1312 | 59792 | 1 | 240x192x128_0 |
| 13. Very Large GEMMs | 123 | 11840 | 8576 | 11360 | 1 | 256x240x128_0 |
| 13. Very Large GEMMs | 46 | 11312 | 12368 | 8928 | 1 | 256x256x128_0 |
| 13. Very Large GEMMs | 134 | 8224 | 11424 | 14160 | 1 | 256x256x128_0 |
| 14. M = 1 | 195 | 1 | 7632 | 36352 | 1 | 16x256x256_0 |
| 14. M = 1 | 93 | 1 | 615664 | 928 | 1 | 16x256x128_0 |
| 14. M = 1 | 88 | 1 | 107408 | 5968 | 1 | 16x256x256_0 |
| 15. N = 1 | 124 | 42720 | 1 | 18784 | 1 | 192x16x256_0 |
| 15. N = 1 | 78 | 2240 | 1 | 922208 | 1 | 128x16x128_0 |
| 15. N = 1 | 58 | 5040 | 1 | 571264 | 1 | 192x16x128_0 |
| 16. K = 1 | 26 | 12912 | 36976 | 1 | 1 | 224x192x32_0 |
| 16. K = 1 | 75 | 640 | 1020544 | 1 | 1 | 160x224x32_0 |
| 16. K = 1 | 200 | 172256 | 5152 | 1 | 1 | 192x224x32_0 |
| 17. Small batch | 126 | 4928 | 6656 | 16 | 40 | 256x224x32_0 |
| 17. Small batch | 50 | 3264 | 2416 | 400 | 7 | 256x224x64_0 |
| 17. Small batch | 161 | 1392 | 4656 | 5488 | 13 | 208x256x128_0 |
| 18. Medium batch | 135 | 1120 | 656 | 512 | 382 | 224x224x64_0 |
| 18. Medium batch | 87 | 448 | 6464 | 448 | 214 | 224x256x64_0 |
| 18. Medium batch | 149 | 96 | 1744 | 6960 | 353 | 96x256x64_0 |
| 19. Large batch | 82 | 48 | 720 | 208 | 3028 | 48x192x64_0 |
| 19. Large batch | 81 | 1776 | 16 | 784 | 4186 | 224x32x64_0 |
| 19. Large batch | 31 | 1488 | 1568 | 16 | 2773 | 256x224x32_0 |
| 20. Very large batch | 90 | 4048 | 16 | 16 | 51732 | 256x16x64_0 |
| 20. Very large batch | 163 | 80 | 3616 | 16 | 23009 | 80x256x32_0 |
| 20. Very large batch | 113 | 2768 | 32 | 208 | 8803 | 160x32x64_0 |
