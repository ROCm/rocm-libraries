Ratios to `tp4-ep-ds1-m0-i8` at the same CONC (dec = per-request median decode tok/s; agg = aggregate output tok/s).

| arm | c1 dec | c2 dec | c4 dec | c8 dec | c16 dec | c64 dec | c1 agg | c2 agg | c4 agg | c8 agg | c16 agg | c64 agg | AL c1 | AL c4 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| tp2-noep-ds0-m0-i8 | 0.306 | 0.352 | 0.367 | 0.433 | 0.492 | - | 0.370 | 0.395 | 0.489 | 0.410 | 0.474 | - | - | - |
| tp2-noep-ds0-m1-i8 | 0.580 | 0.680 | 0.580 | 0.623 | 0.595 | - | 0.657 | 0.707 | 0.661 | 0.756 | 0.641 | - | - | - |
| tp2-noep-ds1-m0-i8 | 0.910 | 0.980 | 0.842 | 0.867 | 0.788 | - | 0.946 | 0.960 | 0.904 | 0.930 | 0.773 | - | 3.76 | 3.80 |
| tp2-noep-ds1-m0-i8-b | 0.915 | 0.964 | 0.852 | 0.934 | 0.845 | - | 0.924 | 0.997 | 0.925 | 0.945 | 0.887 | - | 3.68 | 3.79 |
| tp2-noep-ds1-m1-i4 | 1.435 | 1.468 | 1.042 | 0.867 | 0.821 | - | 1.371 | 1.264 | 1.032 | 0.948 | 0.890 | - | 3.69 | 3.77 |
| tp2-noep-ds1-m1-i8 | 1.444 | 1.386 | 1.011 | 0.850 | 0.808 | - | 1.401 | 1.307 | 1.013 | 0.903 | 0.879 | - | 3.70 | 3.78 |
| tp4-ep-ds0-m0-i8 | 0.313 | 0.384 | 0.401 | 0.475 | 0.592 | 0.595 | 0.383 | 0.406 | 0.446 | 0.436 | 0.547 | 0.542 | - | - |
| tp4-ep-ds1-m0-i8 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 3.79 | 3.88 |
| tp4-ep-ds1-m1-i8 | 0.958 | 1.148 | 1.073 | 1.029 | 1.029 | 0.849 | 0.972 | 1.030 | 1.009 | 0.975 | 1.075 | 0.917 | 3.58 | 3.73 |
| tp4-noep-ds0-m0-i8 | 0.324 | 0.396 | 0.394 | 0.479 | 0.575 | 0.505 | 0.394 | 0.429 | 0.437 | 0.456 | 0.557 | 0.504 | - | - |
| tp4-noep-ds0-m1-i8 | 0.653 | 0.793 | 0.688 | 0.706 | 0.740 | 0.576 | 0.731 | 0.776 | 0.790 | 0.736 | 0.784 | 0.639 | - | - |
| tp4-noep-ds1-m0-i4 | 0.974 | 1.195 | 1.105 | 1.033 | 1.093 | 0.926 | 1.003 | 1.053 | 1.114 | 1.035 | 1.041 | 0.939 | 3.68 | 3.85 |
| tp4-noep-ds1-m0-i8 | 0.999 | 1.111 | 1.073 | 0.976 | 1.004 | 0.734 | 0.987 | 1.052 | 1.083 | 0.871 | 0.978 | 0.849 | 3.71 | 3.82 |
| tp4-noep-ds1-m0-i8-b | 0.983 | 1.187 | 1.107 | 1.079 | 0.973 | 0.847 | 0.993 | 1.004 | 1.051 | 1.009 | 1.041 | 0.900 | 3.66 | 3.69 |
| tp4-noep-ds1-m1-i4 | 1.778 | 1.779 | 1.301 | 1.213 | 0.960 | 0.884 | 1.600 | 1.499 | 1.335 | 1.175 | 1.094 | 0.895 | 3.65 | 3.77 |
| tp4-noep-ds1-m1-i8 | 1.790 | 1.923 | 1.238 | 1.089 | 0.936 | 0.902 | 1.578 | 1.487 | 1.269 | 1.293 | 0.943 | 0.940 | 3.71 | 3.81 |
| tp4-noep-ds1-m1-i8-b | 1.795 | 1.751 | 1.244 | 1.105 | 0.998 | 0.972 | 1.585 | 1.576 | 1.261 | 1.110 | 1.061 | 0.984 | 3.78 | 3.87 |

A-vs-A twins (decode tok/s/req median ratio, twin / original):
- tp2-noep-ds1-m0-i8-b: c1 1.006, c2 0.984, c4 1.012, c8 1.078, c16 1.072
- tp4-noep-ds1-m0-i8-b: c1 0.984, c2 1.069, c4 1.032, c8 1.106, c16 0.969, c64 1.154
- tp4-noep-ds1-m1-i8-b: c1 1.003, c2 0.911, c4 1.005, c8 1.014, c16 1.066, c64 1.077

GSM8K (lm_eval, accuracy):
- tp2-noep-ds1-m0-i8: {"exact_match,strict-match": 0.22, "exact_match,flexible-extract": 0.94}
- tp2-noep-ds1-m1-i8: {"exact_match,strict-match": 0.2, "exact_match,flexible-extract": 0.94}
- tp4-ep-ds1-m0-i8: {"exact_match,strict-match": 0.2, "exact_match,flexible-extract": 0.94}
- tp4-noep-ds1-m0-i8: {"exact_match,strict-match": 0.15, "exact_match,flexible-extract": 0.93}
- tp4-noep-ds1-m1-i8: {"exact_match,strict-match": 0.21, "exact_match,flexible-extract": 0.94}

Mono log lines per arm-run:
- tp2-noep-ds0-m0-i8: {"V4.1 mono decode off: switched off": 2}
- tp2-noep-ds0-m1-i8: {"V4.1 mono decode on": 2}
- tp2-noep-ds1-m0-i8: {"V4.1 mono decode off: switched off": 2}
- tp2-noep-ds1-m0-i8-b: {"V4.1 mono decode off: switched off": 2}
- tp2-noep-ds1-m1-i4: {"V4.1 mono draft on": 2, "V4.1 mono decode on": 2}
- tp2-noep-ds1-m1-i8: {"V4.1 mono draft on": 2, "V4.1 mono decode on": 2}
- tp4-ep-ds0-m0-i8: {"V4.1 mono decode off: switched off": 4}
- tp4-ep-ds1-m0-i8: {"V4.1 mono decode off: switched off": 4}
- tp4-ep-ds1-m1-i8: {"V4.1 mono decode off: expert parallel": 4}
- tp4-noep-ds0-m0-i8: {"V4.1 mono decode off: switched off": 4}
- tp4-noep-ds0-m1-i8: {"V4.1 mono decode on": 4}
- tp4-noep-ds1-m0-i4: {"V4.1 mono decode off: switched off": 4}
- tp4-noep-ds1-m0-i8: {"V4.1 mono decode off: switched off": 4}
- tp4-noep-ds1-m0-i8-b: {"V4.1 mono decode off: switched off": 4}
- tp4-noep-ds1-m1-i4: {"V4.1 mono draft on": 4, "V4.1 mono decode on": 4}
- tp4-noep-ds1-m1-i8: {"V4.1 mono draft on": 4, "V4.1 mono decode on": 4}
- tp4-noep-ds1-m1-i8-b: {"V4.1 mono draft on": 4, "V4.1 mono decode on": 4}
