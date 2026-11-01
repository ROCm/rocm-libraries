# WP-L2a tables (ratios)

Generated from the runs' JSON by `hackathon-prep/qwen38/truemono-lb/wp-l2a/{picks_report,l2a_report}.py`; absolute
numbers in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2a.md`. See `WP-L2a.md` for method and caveats.

## 1. Final picks: four confirm runs (venv, two per node)

Runs: rP1a (smci355-ccs-aus-n01-09 bus 0000:75:00); rP2a (smci355-ccs-aus-n01-09 bus 0000:15:00); rP1b (smci355-ccs-aus-n07-17 bus 0000:65:00); rP2b (smci355-ccs-aus-n07-17 bus 0000:65:00)

The plan table's pick (`e1_plans.PLANS`) per cell, confirm rounds only (24 shuffled rounds, numerics first, second-graph and recompiled controls on every side). Verdicts per run joined (a cell verdict needs every run to agree; else MIXED). `fastest replica` = the fastest of FlyTape's plan on rocKE with P-L3's barrier / the persist barrier, with predicated / branch-free staging. FT in-situ: FlyTape's own traced phase per layer; for out-proj the stricter of gdn_out / qsa_out. Phase metric unless stated.

| cell | pick | pick/base phase | v | pick/base stream | pick/base launch | pick/fastest replica phase | v | pick/ft phase | floor/pick (stream) | pick phase / FT in-situ (same job) | pick phase / FT in-situ (W4 trace) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1_r128 | mt=2 tb=4 depth=4 bar=persist map=xcd sb=False xpad=32 | 0.696 | WIN | 0.933 | 0.914 | 0.280 | WIN | 0.246 | 0.269 | - | - |
| gdn_in_tp1_r16 | kw=4 mt=4 depth=4 bar=persist wcp=nt sb=False xjunk=True | 0.571 | WIN | 0.743 | 0.942 | 0.254 | WIN | 0.206 | 0.561 | - | 0.240 |
| gdn_in_tp1_r32 | kw=2 mt=2 depth=4 bar=persist xa=2 xjunk=True | 0.590 | WIN | 0.902 | 1.044 | 0.273 | WIN | 0.223 | 0.448 | - | 0.249 |
| gdn_in_tp1_r5 | kw=4 mt=4 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.561 | WIN | 0.688 | 0.932 | 0.249 | WIN | 0.203 | 0.613 | - | - |
| gdn_in_tp1_r64 | waves=4 mt=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xpad=32 | 0.639 | WIN | 0.861 | 0.832 | 0.257 | WIN | 0.217 | 0.434 | - | 0.216 |
| gdn_in_tp1_r8 | kw=4 mt=4 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.565 | WIN | 0.706 | 0.934 | 0.252 | WIN | 0.205 | 0.595 | - | 0.234 |
| gdn_in_tp2_r128 | waves=4 mt=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.547 | WIN | 0.718 | 0.762 | 0.268 | WIN | 0.231 | 0.229 | - | 0.323 |
| gdn_in_tp2_r16 | waves=4 kw=4 mt=4 xsrc=global bar=persist wcp=nt sb=False | 0.424 | WIN | 0.584 | 0.757 | 0.271 | WIN | 0.201 | 0.460 | - | 0.360 |
| gdn_in_tp2_r32 | kw=4 mt=2 depth=4 bar=persist xa=2 bpre=2 sb=False xjunk=True | 0.483 | WIN | 0.704 | 0.940 | 0.296 | WIN | 0.221 | 0.358 | - | 0.391 |
| gdn_in_tp2_r5 | waves=4 kw=4 mt=4 depth=4 xsrc=global bar=persist map=xcd wcp=nt sb=False | 0.400 | WIN | 0.593 | 0.564 | 0.259 | WIN | 0.204 | 0.529 | - | - |
| gdn_in_tp2_r64 | kw=2 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.592 | WIN | 0.942 | 1.050 | 0.348 | WIN | 0.280 | 0.236 | - | 0.422 |
| gdn_in_tp2_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.406 | WIN | 0.555 | 0.551 | 0.263 | WIN | 0.199 | 0.513 | - | 0.354 |
| gdn_out_tp1_r128 | kw=4 tb=4 depth=4 bar=persist map=xcd xu=False bpre=2 sb=False xjunk=True xpad=32 | 0.589 | WIN | 0.661 | 0.748 | 0.569 | WIN | 0.458 | 0.154 | - | - |
| gdn_out_tp1_r16 | kw=8 depth=4 bar=persist wcp=nt xa=2 xjunk=True xpad=32 | 0.469 | WIN | 0.603 | 0.817 | 0.485 | WIN | 0.335 | 0.376 | - | 0.415 |
| gdn_out_tp1_r32 | waves=4 kw=4 mt=2 tb=1 depth=4 bar=persist xa=2 bpre=1 sb=False xjunk=True xpad=32 | 0.459 | WIN | 0.522 | 0.682 | 0.478 | WIN | 0.316 | 0.345 | - | 0.440 |
| gdn_out_tp1_r5 | waves=4 kw=4 depth=3 xsrc=global bar=persist map=xcd sb=False | 0.460 | WIN | 0.488 | 0.604 | 0.438 | WIN | 0.315 | 0.474 | - | - |
| gdn_out_tp1_r64 | waves=4 splits=6 depth=4 bar=persist xa=2 sb=False | 0.582 | WIN | 0.877 | 0.931 | 0.634 | WIN | 0.471 | 0.183 | - | 0.566 |
| gdn_out_tp1_r8 | waves=4 kw=4 depth=3 xsrc=global bar=persist map=xcd sb=False | 0.450 | WIN | 0.496 | 0.635 | 0.455 | WIN | 0.324 | 0.433 | - | 0.378 |
| gdn_out_tp2_r128 | kw=4 tb=4 depth=6 bar=persist map=xcd xu=False sb=False xjunk=True xpad=32 | 0.499 | WIN | 0.504 | 0.603 | 0.534 | WIN | 0.404 | 0.120 | - | 0.793 |
| gdn_out_tp2_r16 | kw=8 depth=4 bar=persist map=xcd sb=False xjunk=True | 0.435 | WIN | 0.543 | 0.742 | 0.490 | WIN | 0.320 | 0.276 | - | 0.851 |
| gdn_out_tp2_r32 | kw=8 depth=4 bar=persist xu=False sb=False xjunk=True | 0.454 | WIN | 0.467 | 0.576 | 0.536 | WIN | 0.348 | 0.231 | - | 0.907 |
| gdn_out_tp2_r5 | waves=4 kw=4 xsrc=global bar=persist map=xcd bpre=1 sb=False | 0.409 | WIN | 0.440 | 0.570 | 0.452 | WIN | 0.298 | 0.370 | - | - |
| gdn_out_tp2_r64 | waves=4 kw=4 mt=2 tb=2 depth=3 bar=persist map=xcd xu=False bpre=2 sb=False xjunk=True | 0.441 | WIN | 0.370 | 0.537 | 0.524 | WIN | 0.370 | 0.182 | - | 0.789 |
| gdn_out_tp2_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.425 | WIN | 0.466 | 0.619 | 0.474 | WIN | 0.307 | 0.339 | - | 0.837 |
| hc_down_tpx_r128 | waves=4 splits=5 tb=2 depth=3 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.590 | WIN | 0.684 | 0.767 | 0.622 | WIN | 0.452 | 0.055 | - | 0.725 |
| hc_down_tpx_r16 | waves=4 splits=10 depth=4 bar=persist xu=False wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.583 | WIN | 0.587 | 0.797 | 0.631 | WIN | 0.434 | 0.083 | - | 0.730 |
| hc_down_tpx_r32 | waves=4 splits=10 tb=1 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.589 | WIN | 0.600 | 0.821 | 0.598 | WIN | 0.434 | 0.082 | - | 0.760 |
| hc_down_tpx_r5 | waves=4 splits=20 depth=4 bar=persist xu=False | 0.670 | WIN | 0.934 | 0.930 | 0.734 | WIN | 0.491 | 0.092 | - | - |
| hc_down_tpx_r64 | waves=4 splits=10 tb=1 bar=persist xa=2 fg=16 xpad=32 | 0.573 | WIN | 0.425 | 0.713 | 0.489 | WIN | 0.381 | 0.075 | - | 0.721 |
| hc_down_tpx_r8 | waves=4 splits=10 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.606 | WIN | 0.928 | 0.957 | 0.669 | WIN | 0.455 | 0.090 | - | 0.755 |
| hc_up_tpx_r128 | tb=4 depth=4 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.503 | WIN | 0.668 | 0.571 | 0.672 | WIN | 0.389 | 0.108 | - | 0.527 |
| hc_up_tpx_r16 | waves=4 xsrc=global bar=persist xu=False bpre=1 | 0.455 | WIN | 0.523 | 0.675 | 0.683 | WIN | 0.332 | 0.378 | - | 0.573 |
| hc_up_tpx_r32 | waves=4 depth=6 bar=persist xu=False wcp=nt xjunk=True | 0.498 | WIN | 0.739 | 0.661 | 0.722 | WIN | 0.357 | 0.241 | - | 0.560 |
| hc_up_tpx_r5 | waves=4 depth=3 xsrc=global bar=persist xu=False bpre=1 | 0.439 | WIN | 0.527 | 0.671 | 0.683 | WIN | 0.356 | 0.377 | - | - |
| hc_up_tpx_r64 | waves=4 mt=2 tb=2 depth=4 xsrc=global bar=persist xu=False | 0.466 | WIN | 0.692 | 0.651 | 0.701 | WIN | 0.379 | 0.170 | - | 0.507 |
| hc_up_tpx_r8 | waves=4 xsrc=global bar=persist xu=False wcp=nt | 0.453 | WIN | 0.522 | 0.671 | 0.681 | WIN | 0.345 | 0.379 | - | 0.591 |
| head_tp1_r128 | mt=2 depth=3 bar=persist bpre=2 xjunk=True xpad=32 | 0.915 | WIN | 0.996 | 0.999 | 0.303 | WIN | 0.299 | 0.513 | - | - |
| head_tp1_r16 | mt=4 depth=4 bar=persist map=xcd xu=False wcp=nt xjunk=True | 0.701 | WIN | 0.744 | 0.709 | 0.244 | WIN | 0.238 | 0.879 | - | 0.269 |
| head_tp1_r32 | mt=4 depth=4 bar=persist wcp=nt xa=2 xjunk=True xpad=32 | 0.851 | WIN | 0.939 | 0.905 | 0.262 | WIN | 0.255 | 0.819 | - | 0.279 |
| head_tp1_r5 | mt=4 depth=4 xsrc=global bar=persist xu=False wcp=nt | 0.637 | WIN | 0.664 | 0.644 | 0.218 | WIN | 0.214 | 1.011 | - | - |
| head_tp1_r64 | mt=4 depth=4 bar=persist map=xcd wcp=nt xjunk=True xpad=32 | 0.839 | WIN | 0.925 | 0.887 | 0.273 | WIN | 0.267 | 0.743 | - | 0.329 |
| head_tp1_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt | 0.663 | WIN | 0.703 | 0.669 | 0.232 | WIN | 0.227 | 0.935 | - | 0.257 |
| head_tp2_r128 | mt=2 bar=persist wcp=nt xpad=32 | 0.820 | WIN | 0.995 | 0.982 | 0.303 | WIN | 0.290 | 0.515 | - | 0.371 |
| head_tp2_r16 | mt=4 depth=4 bar=persist wcp=nt xjunk=True | 0.667 | WIN | 0.734 | 0.747 | 0.256 | WIN | 0.239 | 0.892 | - | 0.257 |
| head_tp2_r32 | mt=4 depth=3 bar=persist wcp=nt sb=False xjunk=True | 0.740 | WIN | 0.926 | 0.871 | 0.269 | WIN | 0.250 | 0.831 | - | 0.264 |
| head_tp2_r5 | mt=4 depth=4 xsrc=global bar=persist xu=False wcp=nt bpre=1 | 0.608 | WIN | 0.666 | 0.654 | 0.229 | WIN | 0.217 | 1.013 | - | - |
| head_tp2_r64 | mt=4 depth=3 bar=persist wcp=nt xjunk=True xpad=32 | 0.744 | WIN | 0.908 | 0.888 | 0.282 | WIN | 0.264 | 0.749 | - | 0.302 |
| head_tp2_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt bpre=1 sb=False | 0.630 | WIN | 0.700 | 0.705 | 0.240 | WIN | 0.227 | 0.951 | - | 0.246 |
| qsa_in_tp1_r128 | waves=4 mt=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False | 0.595 | WIN | 0.843 | 0.844 | 0.265 | WIN | 0.210 | 0.368 | - | - |
| qsa_in_tp1_r16 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.443 | WIN | 0.532 | 0.715 | 0.253 | WIN | 0.184 | 0.669 | - | 0.256 |
| qsa_in_tp1_r32 | kw=4 mt=2 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.486 | WIN | 0.626 | 0.836 | 0.290 | WIN | 0.205 | 0.555 | - | 0.289 |
| qsa_in_tp1_r5 | waves=4 kw=4 mt=4 depth=4 xsrc=global bar=persist wcp=nt sb=False | 0.436 | WIN | 0.445 | 0.663 | 0.249 | WIN | 0.181 | 0.811 | - | - |
| qsa_in_tp1_r64 | kw=4 mt=2 depth=3 bar=persist map=xcd wcp=nt sb=False | 0.644 | WIN | 0.875 | 0.909 | 0.335 | WIN | 0.247 | 0.368 | - | 0.306 |
| qsa_in_tp1_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt | 0.451 | WIN | 0.483 | 0.675 | 0.258 | WIN | 0.186 | 0.744 | - | 0.258 |
| qsa_in_tp2_r128 | kw=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.536 | WIN | 0.686 | 0.798 | 0.401 | WIN | 0.287 | 0.212 | - | 0.418 |
| qsa_in_tp2_r16 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.356 | WIN | 0.443 | 0.479 | 0.299 | WIN | 0.198 | 0.563 | - | 0.331 |
| qsa_in_tp2_r32 | waves=4 kw=4 mt=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xjunk=True | 0.413 | WIN | 0.511 | 0.716 | 0.357 | WIN | 0.237 | 0.437 | - | 0.381 |
| qsa_in_tp2_r5 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.353 | WIN | 0.409 | 0.457 | 0.292 | WIN | 0.220 | 0.679 | - | - |
| qsa_in_tp2_r64 | waves=4 kw=4 mt=2 depth=4 bar=persist sb=False xjunk=True | 0.483 | WIN | 0.504 | 0.616 | 0.415 | WIN | 0.307 | 0.289 | - | 0.434 |
| qsa_in_tp2_r8 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.355 | WIN | 0.405 | 0.464 | 0.291 | WIN | 0.211 | 0.631 | - | 0.313 |
| router_tpx_r128 | waves=4 kw=4 tb=2 depth=4 bar=persist xu=False xa=2 xjunk=True | 0.421 | WIN | 0.500 | 0.575 | 0.314 | WIN | 0.259 | 0.049 | - | 0.244 |
| router_tpx_r16 | waves=4 kw=4 depth=4 bar=persist xu=False sb=False xjunk=True xpad=32 | 0.386 | WIN | 0.494 | 0.653 | 0.510 | WIN | 0.335 | 0.069 | - | 0.216 |
| router_tpx_r32 | waves=4 kw=4 tb=1 depth=4 bar=persist xu=False sb=False xjunk=True | 0.359 | WIN | 0.489 | 0.643 | 0.435 | WIN | 0.307 | 0.070 | - | 0.222 |
| router_tpx_r5 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.390 | WIN | 0.584 | 0.749 | 0.507 | WIN | 0.328 | 0.085 | - | - |
| router_tpx_r64 | waves=4 kw=4 tb=1 xsrc=global bar=persist sb=False | 0.391 | WIN | 0.492 | 0.704 | 0.390 | WIN | 0.292 | 0.066 | - | 0.229 |
| router_tpx_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.398 | WIN | 0.631 | 0.786 | 0.519 | WIN | 0.338 | 0.077 | - | 0.222 |

Verdict tallies (cells): pick vs base: WIN 66; pick vs fastest replica: WIN 66

### Per projection (ranges over rows 5-128 of the per-cell medians)

| projection | pick/base phase | pick/base stream | pick/base launch | pick/fastest replica phase | pick/ft phase | floor/pick (stream) | pick / FT in-situ (same job) | pick / FT in-situ (W4 trace) |
|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1 | 0.56-0.70 | 0.69-0.93 | 0.83-1.04 | 0.25-0.28 | 0.20-0.25 | 0.27-0.61 | - | 0.22-0.25 |
| gdn_in_tp2 | 0.40-0.59 | 0.56-0.94 | 0.55-1.05 | 0.26-0.35 | 0.20-0.28 | 0.23-0.53 | - | 0.32-0.42 |
| gdn_out_tp1 | 0.45-0.59 | 0.49-0.88 | 0.60-0.93 | 0.44-0.63 | 0.32-0.47 | 0.15-0.47 | - | 0.38-0.57 |
| gdn_out_tp2 | 0.41-0.50 | 0.37-0.54 | 0.54-0.74 | 0.45-0.54 | 0.30-0.40 | 0.12-0.37 | - | 0.79-0.91 |
| hc_down_tpx | 0.57-0.67 | 0.42-0.93 | 0.71-0.96 | 0.49-0.73 | 0.38-0.49 | 0.06-0.09 | - | 0.72-0.76 |
| hc_up_tpx | 0.44-0.50 | 0.52-0.74 | 0.57-0.68 | 0.67-0.72 | 0.33-0.39 | 0.11-0.38 | - | 0.51-0.59 |
| head_tp1 | 0.64-0.92 | 0.66-1.00 | 0.64-1.00 | 0.22-0.30 | 0.21-0.30 | 0.51-1.01 | - | 0.26-0.33 |
| head_tp2 | 0.61-0.82 | 0.67-0.99 | 0.65-0.98 | 0.23-0.30 | 0.22-0.29 | 0.52-1.01 | - | 0.25-0.37 |
| qsa_in_tp1 | 0.44-0.64 | 0.45-0.88 | 0.66-0.91 | 0.25-0.34 | 0.18-0.25 | 0.37-0.81 | - | 0.26-0.31 |
| qsa_in_tp2 | 0.35-0.54 | 0.41-0.69 | 0.46-0.80 | 0.29-0.41 | 0.20-0.31 | 0.21-0.68 | - | 0.31-0.43 |
| router_tpx | 0.36-0.42 | 0.49-0.63 | 0.57-0.79 | 0.31-0.52 | 0.26-0.34 | 0.05-0.09 | - | 0.22-0.24 |

Numerics of these runs: per-cell checks 3168 (failing 0); suite records 436 (mutants caught 332 / 332); chain checks 96; barrier timeouts 0.


## 2. Final picks: the container run with FlyTape traced in the same job (before and after the LB timing; the ROCm 10 image's COMGR)

Runs: C2 (smci355-ccs-aus-n01-09 bus 0000:75:00)

The plan table's pick (`e1_plans.PLANS`) per cell, confirm rounds only (24 shuffled rounds, numerics first, second-graph and recompiled controls on every side). Verdicts per run joined (a cell verdict needs every run to agree; else MIXED). `fastest replica` = the fastest of FlyTape's plan on rocKE with P-L3's barrier / the persist barrier, with predicated / branch-free staging. FT in-situ: FlyTape's own traced phase per layer; for out-proj the stricter of gdn_out / qsa_out. Phase metric unless stated.

| cell | pick | pick/base phase | v | pick/base stream | pick/base launch | pick/fastest replica phase | v | pick/ft phase | floor/pick (stream) | pick phase / FT in-situ (same job) | pick phase / FT in-situ (W4 trace) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1_r16 | kw=4 mt=4 depth=4 bar=persist wcp=nt sb=False xjunk=True | 0.575 | WIN | 0.839 | 1.053 | 0.265 | WIN | 0.212 | 0.588 | 0.232 | 0.232 |
| gdn_in_tp1_r32 | kw=2 mt=2 depth=4 bar=persist xa=2 xjunk=True | 0.601 | WIN | 0.908 | 1.043 | 0.277 | WIN | 0.228 | 0.470 | 0.247 | 0.247 |
| gdn_in_tp1_r64 | waves=4 mt=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xpad=32 | 0.669 | WIN | 0.881 | 0.859 | 0.269 | WIN | 0.225 | 0.439 | 0.221 | 0.222 |
| gdn_in_tp1_r8 | kw=4 mt=4 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.566 | WIN | 0.802 | 1.059 | 0.257 | WIN | 0.208 | 0.623 | 0.231 | 0.226 |
| gdn_in_tp2_r128 | waves=4 mt=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.572 | WIN | 0.759 | 0.803 | 0.281 | WIN | 0.243 | 0.224 | 0.341 | 0.339 |
| gdn_in_tp2_r16 | waves=4 kw=4 mt=4 xsrc=global bar=persist wcp=nt sb=False | 0.417 | WIN | 0.599 | 0.777 | 0.279 | WIN | 0.218 | 0.486 | 0.352 | 0.351 |
| gdn_in_tp2_r32 | kw=4 mt=2 depth=4 bar=persist xa=2 bpre=2 sb=False xjunk=True | 0.505 | WIN | 0.752 | 0.961 | 0.314 | WIN | 0.236 | 0.355 | 0.404 | 0.403 |
| gdn_in_tp2_r64 | kw=2 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.606 | WIN | 0.941 | 1.041 | 0.367 | WIN | 0.288 | 0.246 | 0.440 | 0.434 |
| gdn_in_tp2_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.423 | WIN | 0.564 | 0.524 | 0.284 | WIN | 0.228 | 0.521 | 0.373 | 0.360 |
| gdn_out_tp1_r16 | kw=8 depth=4 bar=persist wcp=nt xa=2 xjunk=True xpad=32 | 0.466 | WIN | 0.578 | 0.785 | 0.522 | WIN | 0.364 | 0.399 | 0.409 | 0.411 |
| gdn_out_tp1_r32 | waves=4 kw=4 mt=2 tb=1 depth=4 bar=persist xa=2 bpre=1 sb=False xjunk=True xpad=32 | 0.448 | WIN | 0.504 | 0.654 | 0.496 | WIN | 0.353 | 0.368 | 0.426 | 0.433 |
| gdn_out_tp1_r64 | waves=4 splits=6 depth=4 bar=persist xa=2 sb=False | 0.588 | WIN | 0.861 | 0.929 | 0.659 | WIN | 0.499 | 0.194 | 0.556 | 0.562 |
| gdn_out_tp1_r8 | waves=4 kw=4 depth=3 xsrc=global bar=persist map=xcd sb=False | 0.445 | WIN | 0.448 | 0.614 | 0.473 | WIN | 0.330 | 0.463 | 0.376 | 0.375 |
| gdn_out_tp2_r128 | kw=4 tb=4 depth=6 bar=persist map=xcd xu=False sb=False xjunk=True xpad=32 | 0.499 | WIN | 0.520 | 0.617 | 0.549 | WIN | 0.416 | 0.121 | 0.833 | 0.811 |
| gdn_out_tp2_r16 | kw=8 depth=4 bar=persist map=xcd sb=False xjunk=True | 0.430 | WIN | 0.478 | 0.726 | 0.517 | WIN | 0.335 | 0.294 | 0.854 | 0.850 |
| gdn_out_tp2_r32 | kw=8 depth=4 bar=persist xu=False sb=False xjunk=True | 0.459 | WIN | 0.480 | 0.586 | 0.555 | WIN | 0.374 | 0.240 | 0.927 | 0.946 |
| gdn_out_tp2_r64 | waves=4 kw=4 mt=2 tb=2 depth=3 bar=persist map=xcd xu=False bpre=2 sb=False xjunk=True | 0.433 | WIN | 0.394 | 0.557 | 0.541 | WIN | 0.379 | 0.181 | 0.867 | 0.794 |
| gdn_out_tp2_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.433 | WIN | 0.483 | 0.651 | 0.524 | WIN | 0.328 | 0.345 | 0.878 | 0.858 |
| hc_down_tpx_r128 | waves=4 splits=5 tb=2 depth=3 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.612 | WIN | 0.701 | 0.752 | 0.634 | WIN | 0.455 | 0.055 | 0.727 | 0.734 |
| hc_down_tpx_r16 | waves=4 splits=10 depth=4 bar=persist xu=False wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.597 | WIN | 0.602 | 0.813 | 0.663 | WIN | 0.462 | 0.084 | 0.746 | 0.746 |
| hc_down_tpx_r32 | waves=4 splits=10 tb=1 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.597 | WIN | 0.596 | 0.815 | 0.621 | WIN | 0.435 | 0.084 | 0.767 | 0.767 |
| hc_down_tpx_r64 | waves=4 splits=10 tb=1 bar=persist xa=2 fg=16 xpad=32 | 0.590 | WIN | 0.401 | 0.707 | 0.517 | WIN | 0.395 | 0.079 | 0.736 | 0.738 |
| hc_down_tpx_r8 | waves=4 splits=10 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.603 | WIN | 0.911 | 0.930 | 0.682 | WIN | 0.455 | 0.095 | 0.742 | 0.746 |
| hc_up_tpx_r128 | tb=4 depth=4 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.514 | WIN | 0.664 | 0.577 | 0.690 | WIN | 0.381 | 0.114 | 0.516 | 0.522 |
| hc_up_tpx_r16 | waves=4 xsrc=global bar=persist xu=False bpre=1 | 0.466 | WIN | 0.500 | 0.669 | 0.719 | WIN | 0.347 | 0.396 | 0.595 | 0.598 |
| hc_up_tpx_r32 | waves=4 depth=6 bar=persist xu=False wcp=nt xjunk=True | 0.515 | WIN | 0.689 | 0.678 | 0.729 | WIN | 0.367 | 0.255 | 0.578 | 0.579 |
| hc_up_tpx_r64 | waves=4 mt=2 tb=2 depth=4 xsrc=global bar=persist xu=False | 0.493 | WIN | 0.605 | 0.642 | 0.739 | WIN | 0.391 | 0.184 | 0.533 | 0.534 |
| hc_up_tpx_r8 | waves=4 xsrc=global bar=persist xu=False wcp=nt | 0.453 | WIN | 0.523 | 0.661 | 0.723 | WIN | 0.359 | 0.383 | 0.613 | 0.615 |
| head_tp1_r16 | mt=4 depth=4 bar=persist map=xcd xu=False wcp=nt xjunk=True | 0.724 | WIN | 0.757 | 0.735 | 0.257 | WIN | 0.250 | 0.938 | 0.265 | 0.264 |
| head_tp1_r32 | mt=4 depth=4 bar=persist wcp=nt xa=2 xjunk=True xpad=32 | 0.851 | WIN | 0.943 | 0.898 | 0.242 | WIN | 0.235 | 0.874 | 0.275 | 0.267 |
| head_tp1_r64 | mt=4 depth=4 bar=persist map=xcd wcp=nt xjunk=True xpad=32 | 0.846 | WIN | 0.931 | 0.884 | 0.256 | WIN | 0.251 | 0.775 | 0.322 | 0.321 |
| head_tp1_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt | 0.678 | WIN | 0.723 | 0.702 | 0.234 | WIN | 0.229 | 1.007 | 0.248 | 0.246 |
| head_tp2_r128 | mt=2 bar=persist wcp=nt xpad=32 | 0.822 | WIN | 0.998 | 0.981 | 0.290 | WIN | 0.278 | 0.534 | 0.363 | 0.365 |
| head_tp2_r16 | mt=4 depth=4 bar=persist wcp=nt xjunk=True | 0.675 | WIN | 0.751 | 0.769 | 0.266 | WIN | 0.249 | 0.957 | 0.270 | 0.249 |
| head_tp2_r32 | mt=4 depth=3 bar=persist wcp=nt sb=False xjunk=True | 0.745 | WIN | 0.938 | 0.947 | 0.242 | WIN | 0.225 | 0.993 | 0.255 | 0.240 |
| head_tp2_r64 | mt=4 depth=3 bar=persist wcp=nt xjunk=True xpad=32 | 0.742 | WIN | 0.884 | 0.915 | 0.234 | WIN | 0.222 | 0.905 | 0.278 | 0.266 |
| head_tp2_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt bpre=1 sb=False | 0.640 | WIN | 0.719 | 0.740 | 0.245 | WIN | 0.231 | 1.021 | 0.257 | 0.237 |
| qsa_in_tp1_r16 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.479 | WIN | 0.621 | 0.822 | 0.286 | WIN | 0.201 | 0.687 | 0.265 | 0.266 |
| qsa_in_tp1_r32 | kw=4 mt=2 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.500 | WIN | 0.663 | 0.854 | 0.296 | WIN | 0.208 | 0.548 | 0.290 | 0.290 |
| qsa_in_tp1_r64 | kw=4 mt=2 depth=3 bar=persist map=xcd wcp=nt sb=False | 0.660 | WIN | 0.867 | 0.923 | 0.346 | WIN | 0.255 | 0.381 | 0.313 | 0.311 |
| qsa_in_tp1_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt | 0.465 | WIN | 0.554 | 0.771 | 0.273 | WIN | 0.190 | 0.780 | 0.253 | 0.253 |
| qsa_in_tp2_r128 | kw=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.539 | WIN | 0.688 | 0.794 | 0.400 | WIN | 0.316 | 0.219 | 0.416 | 0.416 |
| qsa_in_tp2_r16 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.371 | WIN | 0.408 | 0.482 | 0.334 | WIN | 0.234 | 0.574 | 0.346 | 0.346 |
| qsa_in_tp2_r32 | waves=4 kw=4 mt=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xjunk=True | 0.421 | WIN | 0.507 | 0.744 | 0.363 | WIN | 0.256 | 0.466 | 0.379 | 0.379 |
| qsa_in_tp2_r64 | waves=4 kw=4 mt=2 depth=4 bar=persist sb=False xjunk=True | 0.509 | WIN | 0.509 | 0.631 | 0.449 | WIN | 0.329 | 0.296 | 0.467 | 0.462 |
| qsa_in_tp2_r8 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.384 | WIN | 0.431 | 0.446 | 0.320 | WIN | 0.232 | 0.634 | 0.345 | 0.330 |
| router_tpx_r128 | waves=4 kw=4 tb=2 depth=4 bar=persist xu=False xa=2 xjunk=True | 0.421 | WIN | 0.486 | 0.583 | 0.332 | WIN | 0.265 | 0.048 | 0.247 | 0.252 |
| router_tpx_r16 | waves=4 kw=4 depth=4 bar=persist xu=False sb=False xjunk=True xpad=32 | 0.374 | WIN | 0.476 | 0.629 | 0.502 | WIN | 0.329 | 0.073 | 0.213 | 0.212 |
| router_tpx_r32 | waves=4 kw=4 tb=1 depth=4 bar=persist xu=False sb=False xjunk=True | 0.355 | WIN | 0.477 | 0.631 | 0.440 | WIN | 0.300 | 0.072 | 0.219 | 0.219 |
| router_tpx_r64 | waves=4 kw=4 tb=1 xsrc=global bar=persist sb=False | 0.389 | WIN | 0.477 | 0.703 | 0.388 | WIN | 0.287 | 0.069 | 0.226 | 0.226 |
| router_tpx_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.394 | WIN | 0.603 | 0.770 | 0.521 | WIN | 0.331 | 0.080 | 0.222 | 0.220 |

Verdict tallies (cells): pick vs base: WIN 51; pick vs fastest replica: WIN 51

### Per projection (ranges over rows 5-128 of the per-cell medians)

| projection | pick/base phase | pick/base stream | pick/base launch | pick/fastest replica phase | pick/ft phase | floor/pick (stream) | pick / FT in-situ (same job) | pick / FT in-situ (W4 trace) |
|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1 | 0.57-0.67 | 0.80-0.91 | 0.86-1.06 | 0.26-0.28 | 0.21-0.23 | 0.44-0.62 | 0.22-0.25 | 0.22-0.25 |
| gdn_in_tp2 | 0.42-0.61 | 0.56-0.94 | 0.52-1.04 | 0.28-0.37 | 0.22-0.29 | 0.22-0.52 | 0.34-0.44 | 0.34-0.43 |
| gdn_out_tp1 | 0.45-0.59 | 0.45-0.86 | 0.61-0.93 | 0.47-0.66 | 0.33-0.50 | 0.19-0.46 | 0.38-0.56 | 0.38-0.56 |
| gdn_out_tp2 | 0.43-0.50 | 0.39-0.52 | 0.56-0.73 | 0.52-0.56 | 0.33-0.42 | 0.12-0.34 | 0.83-0.93 | 0.79-0.95 |
| hc_down_tpx | 0.59-0.61 | 0.40-0.91 | 0.71-0.93 | 0.52-0.68 | 0.40-0.46 | 0.06-0.10 | 0.73-0.77 | 0.73-0.77 |
| hc_up_tpx | 0.45-0.52 | 0.50-0.69 | 0.58-0.68 | 0.69-0.74 | 0.35-0.39 | 0.11-0.40 | 0.52-0.61 | 0.52-0.61 |
| head_tp1 | 0.68-0.85 | 0.72-0.94 | 0.70-0.90 | 0.23-0.26 | 0.23-0.25 | 0.78-1.01 | 0.25-0.32 | 0.25-0.32 |
| head_tp2 | 0.64-0.82 | 0.72-1.00 | 0.74-0.98 | 0.23-0.29 | 0.22-0.28 | 0.53-1.02 | 0.26-0.36 | 0.24-0.36 |
| qsa_in_tp1 | 0.47-0.66 | 0.55-0.87 | 0.77-0.92 | 0.27-0.35 | 0.19-0.26 | 0.38-0.78 | 0.25-0.31 | 0.25-0.31 |
| qsa_in_tp2 | 0.37-0.54 | 0.41-0.69 | 0.45-0.79 | 0.32-0.45 | 0.23-0.33 | 0.22-0.63 | 0.34-0.47 | 0.33-0.46 |
| router_tpx | 0.35-0.42 | 0.48-0.60 | 0.58-0.77 | 0.33-0.52 | 0.27-0.33 | 0.05-0.08 | 0.21-0.25 | 0.21-0.25 |
Same-job FlyTape in-situ: 51 cells, pick phase / FlyTape phase 0.213-0.927; above 1.0: none

Numerics of these runs: per-cell checks 612 (failing 0); suite records 0 (mutants caught 0 / 0); chain checks 0; barrier timeouts 0.


## 3. Library arms (container, launch metric: one call per layer in a graph; LB arm = the fastest fp8 / fp4 finalist of the cell's screen by launch time)

| cell | LB W4A8 / ATOM `native_quant_linear` (W4A8) | v | LB W4A4 (`act=fp4`) / ATOM W4A8 | v | LB W4A8 / aiter `gemm_a4w4` (W4A4) | v | LB W4A4 / aiter `gemm_a4w4` | v | LB fp4 / LB fp8 (launch, phase) |
|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1_r64 | 0.116 | WIN | 0.135 | WIN | 1.330 | LOSS | 1.542 | LOSS | 1.162, 1.076 |
| gdn_in_tp1_r128 | 0.110 | WIN | 0.113 | WIN | 1.606 | LOSS | 1.650 | LOSS | 1.026, 1.037 |
| gdn_in_tp2_r64 | 0.140 | WIN | 0.134 | WIN | 1.301 | LOSS | 1.240 | LOSS | 0.954, 0.915 |
| gdn_in_tp2_r128 | 0.115 | WIN | 0.154 | WIN | 1.193 | LOSS | 1.595 | LOSS | 1.338, 1.252 |
| qsa_in_tp1_r64 | 0.146 | WIN | 0.139 | WIN | 1.320 | LOSS | 1.264 | LOSS | 0.956, 0.932 |
| qsa_in_tp1_r128 | 0.097 | WIN | 0.126 | WIN | 1.209 | LOSS | 1.566 | LOSS | 1.295, 1.280 |
| qsa_in_tp2_r64 | 0.141 | WIN | 0.150 | WIN | 1.023 | LOSS | 1.091 | LOSS | 1.065, 0.912 |
| qsa_in_tp2_r128 | 0.137 | WIN | 0.137 | WIN | 1.139 | LOSS | 1.141 | LOSS | 1.001, 1.064 |
| head_tp1_r64 | 0.080 | WIN | 0.090 | WIN | 0.982 | TIE | 1.105 | LOSS | 1.130, 1.076 |
| head_tp1_r128 | 0.066 | WIN | 0.060 | WIN | 1.202 | LOSS | 1.099 | LOSS | 0.919, 0.930 |
| head_tp2_r64 | 0.076 | WIN | 0.090 | WIN | 0.937 | WIN | 1.125 | LOSS | 1.199, 1.065 |
| head_tp2_r128 | 0.066 | WIN | 0.060 | WIN | 1.315 | LOSS | 1.213 | LOSS | 0.920, 0.956 |

## 4. Lookback climb log (two nodes; a change becomes the pick after a WIN over the previous pick on the second node)

Events: confirmed on n01: 21, confirmed on n07: 13, reverted on n01: 23, reverted on n07: 32, trial on n01: 45, trial on n07: 56

| cell | event | node | lever set (trial) | ratio to the previous pick (phase) |
|---|---|---|---|---|
| ('gdn_out', 2, 5) | trial | n01 | map+bpre | 0.975 |
| ('gdn_out', 2, 64) | trial | n01 | xsrc+bpre+map | 0.884 |
| ('gdn_out', 2, 128) | trial | n01 | noxu+d3+map | 0.989 |
| ('hc_down', 0, 5) | trial | n01 | nt+xsrc | 0.977 |
| ('hc_down', 0, 8) | trial | n01 | s/2+nt | 0.932 |
| ('hc_down', 0, 16) | trial | n01 | w4+nosb+nt | 0.948 |
| ('hc_down', 0, 32) | trial | n01 | s/2+mt/2+noxu | 0.851 |
| ('hc_down', 0, 64) | trial | n01 | s/2+tb/2+w4 | 0.781 |
| ('hc_down', 0, 128) | trial | n01 | kw*2+bpre | 0.948 |
| ('hc_up', 0, 5) | trial | n01 | nt | 0.983 |
| ('hc_up', 0, 8) | trial | n01 | nt+d3 | 0.987 |
| ('hc_up', 0, 32) | trial | n01 | xsrc+d6 | 0.972 |
| ('gdn_out', 2, 8) | trial | n07 | nt+bpre | 0.976 |
| ('gdn_out', 2, 64) | confirmed | n07 |  | 0.897 |
| ('gdn_out', 2, 64) | trial | n07 | d3+nosb+nt | 0.931 |
| ('gdn_out', 2, 128) | reverted | n07 |  | 0.998 |
| ('gdn_out', 1, 5) | trial | n07 | w4+d3+nosb | 0.906 |
| ('gdn_out', 1, 8) | trial | n07 | w4+d3+nosb | 0.924 |
| ('gdn_out', 1, 16) | trial | n07 | mt/2+xpad+nt | 0.859 |
| ('gdn_out', 1, 32) | trial | n07 | tb/2+w4 | 0.758 |
| ('gdn_out', 1, 64) | trial | n07 | nosb | 0.962 |
| ('gdn_out', 1, 128) | trial | n07 | noxu+map+bpre | 0.979 |
| ('hc_down', 0, 5) | reverted | n07 |  | 0.990 |
| ('hc_down', 0, 8) | confirmed | n07 |  | 0.936 |
| ('hc_down', 0, 8) | trial | n07 | xa2+fg16+xpad | 0.970 |
| ('hc_down', 0, 16) | confirmed | n07 |  | 0.952 |
| ('hc_down', 0, 16) | trial | n07 | mt/2+xjunk | 0.865 |
| ('hc_down', 0, 32) | confirmed | n07 |  | 0.876 |
| ('hc_down', 0, 32) | trial | n07 | noxu+xa2 | 0.958 |
| ('hc_down', 0, 64) | confirmed | n07 |  | 0.772 |
| ('hc_down', 0, 64) | trial | n07 | mt/2+d2 | 0.917 |
| ('hc_down', 0, 128) | confirmed | n07 |  | 0.945 |
| ('hc_down', 0, 128) | trial | n07 | s/2+xa2+d3 | 0.958 |
| ('hc_up', 0, 32) | confirmed | n07 |  | 0.981 |
| ('hc_up', 0, 32) | trial | n07 | map+xpad+xjunk | 0.976 |
| ('hc_up', 0, 64) | trial | n07 | dall | 0.985 |
| ('router', 0, 128) | trial | n07 | nosb+d3+xpad | 0.979 |
| ('gdn_in', 2, 5) | trial | n07 | d2+bpre | 0.948 |
| ('gdn_in', 2, 16) | trial | n07 | bpre+nt | 0.969 |
| ('gdn_in', 2, 128) | trial | n07 | w4+nosb+map | 0.889 |
| ('gdn_in', 1, 5) | trial | n07 | map+xpad | 0.976 |
| ('gdn_in', 1, 16) | trial | n07 | xa2+map | 0.960 |
| ('gdn_in', 1, 64) | trial | n07 | w4+xpad+map | 0.923 |
| ('gdn_in', 1, 128) | trial | n07 | map+xpad+xjunk | 0.967 |
| ('qsa_in', 2, 32) | trial | n07 | xa2+nt+map | 0.968 |
| ('qsa_in', 2, 128) | trial | n07 | map+nosb | 0.958 |
| ('qsa_in', 1, 5) | trial | n07 | d2+bpre | 0.968 |
| ('qsa_in', 1, 8) | trial | n07 | nosb+nt+map | 0.948 |
| ('qsa_in', 1, 64) | trial | n07 | d3+map+xjunk | 0.970 |
| ('qsa_in', 1, 128) | trial | n07 | w4+map+noxu | 0.916 |
| ('head', 2, 32) | trial | n07 | w4 | 0.947 |
| ('gdn_out', 2, 8) | reverted | n01 |  | 0.998 |
| ('gdn_out', 2, 32) | trial | n01 | map | 0.972 |
| ('gdn_out', 2, 64) | confirmed | n01 |  | 0.946 |
| ('gdn_out', 2, 64) | trial | n01 | nt+noxu+w4 | 0.960 |
| ('gdn_out', 2, 128) | trial | n01 | noxu+dall+map | 0.933 |
| ('gdn_out', 1, 5) | confirmed | n01 |  | 0.917 |
| ('gdn_out', 1, 8) | confirmed | n01 |  | 0.904 |
| ('gdn_out', 1, 16) | confirmed | n01 |  | 0.887 |
| ('gdn_out', 1, 32) | confirmed | n01 |  | 0.763 |
| ('gdn_out', 1, 32) | trial | n01 | xa2+bpre+xpad | 0.927 |
| ('gdn_out', 1, 64) | confirmed | n01 |  | 0.968 |
| ('gdn_out', 1, 128) | confirmed | n01 |  | 0.983 |
| ('hc_down', 0, 5) | trial | n01 | fg16 | 0.983 |
| ('hc_down', 0, 8) | reverted | n01 |  | 0.975 |
| ('hc_down', 0, 16) | confirmed | n01 |  | 0.888 |
| ('hc_down', 0, 16) | trial | n01 | s/2+xa2 | 0.890 |
| ('hc_down', 0, 32) | confirmed | n01 |  | 0.965 |
| ('hc_down', 0, 32) | trial | n01 | bpre | 0.984 |
| ('hc_down', 0, 64) | confirmed | n01 |  | 0.921 |
| ('hc_down', 0, 64) | trial | n01 | fg16 | 0.992 |
| ('hc_down', 0, 128) | confirmed | n01 |  | 0.960 |
| ('hc_down', 0, 128) | trial | n01 | w4+noxu | 0.974 |
| ('hc_up', 0, 32) | reverted | n01 |  | 0.986 |
| ('hc_up', 0, 64) | reverted | n01 |  | 0.998 |
| ('router', 0, 8) | trial | n01 | nt | 0.961 |
| ('router', 0, 16) | trial | n01 | nt | 0.962 |
| ('router', 0, 128) | reverted | n01 |  | 0.994 |
| ('gdn_in', 2, 5) | reverted | n01 |  | 0.988 |
| ('gdn_in', 2, 8) | trial | n01 | map+d2 | 0.988 |
| ('gdn_in', 2, 16) | reverted | n01 |  | 1.008 |
| ('gdn_in', 2, 128) | confirmed | n01 |  | 0.892 |
| ('gdn_in', 1, 5) | reverted | n01 |  | 0.999 |
| ('gdn_in', 1, 16) | reverted | n01 |  | 0.977 |
| ('gdn_in', 1, 64) | confirmed | n01 |  | 0.932 |
| ('gdn_in', 1, 128) | confirmed | n01 |  | 0.969 |
| ('qsa_in', 2, 32) | confirmed | n01 |  | 0.972 |
| ('qsa_in', 2, 128) | confirmed | n01 |  | 0.961 |
| ('qsa_in', 1, 5) | reverted | n01 |  | 0.997 |
| ('qsa_in', 1, 8) | reverted | n01 |  | 0.987 |
| ('qsa_in', 1, 64) | confirmed | n01 |  | 0.983 |
| ('qsa_in', 1, 128) | confirmed | n01 |  | 0.919 |
| ('head', 2, 5) | trial | n01 | mt/2 | 0.969 |
| ('head', 2, 8) | trial | n01 | w4+d3 | 0.965 |
| ('head', 2, 32) | reverted | n01 |  | 0.990 |
| ('head', 2, 64) | trial | n01 | noxu+nosb+xjunk | 0.979 |
| ('head', 2, 128) | trial | n01 | w4+bpre+nosb | 0.939 |
| ('head', 1, 32) | trial | n01 | nosb+w4 | 0.984 |
| ('head', 1, 128) | trial | n01 | nt+w4+nosb | 0.945 |
| ('gdn_out', 2, 5) | reverted | n07 |  | 0.979 |
| ('gdn_out', 2, 16) | trial | n07 | noxu+nt | 0.973 |
| ('gdn_out', 2, 32) | reverted | n07 |  | 0.991 |
| ('gdn_out', 2, 64) | confirmed | n07 |  | 0.964 |
| ('gdn_out', 2, 128) | confirmed | n07 |  | 0.942 |
| ('gdn_out', 1, 32) | confirmed | n07 |  | 0.950 |
| ('gdn_out', 1, 32) | trial | n07 | map+d2 | 0.979 |
| ('gdn_out', 1, 128) | trial | n07 | d6+nt | 0.980 |
| ('hc_down', 0, 5) | reverted | n07 |  | 1.000 |
| ('hc_down', 0, 8) | trial | n07 | xa2+nt | 0.986 |
| ('hc_down', 0, 16) | confirmed | n07 |  | 0.908 |
| ('hc_down', 0, 16) | trial | n07 | dall+fg16 | 0.990 |
| ('hc_down', 0, 32) | reverted | n07 |  | 1.000 |
| ('hc_down', 0, 64) | reverted | n07 |  | None |
| ('hc_down', 0, 128) | confirmed | n07 |  | 0.982 |
| ('hc_down', 0, 128) | trial | n07 | kw/2+xpad | 0.920 |
| ('hc_up', 0, 5) | reverted | n07 |  | 0.996 |
| ('hc_up', 0, 8) | reverted | n07 |  | 1.005 |
| ('hc_up', 0, 16) | trial | n07 | nt | 0.989 |
| ('hc_up', 0, 128) | trial | n07 | xa2+bpre+nosb | 0.979 |
| ('router', 0, 8) | reverted | n07 |  | 0.998 |
| ('router', 0, 16) | reverted | n07 |  | 0.998 |
| ('router', 0, 32) | trial | n07 | bpre | 0.988 |
| ('gdn_in', 2, 8) | reverted | n07 |  | 1.001 |
| ('gdn_in', 1, 16) | trial | n07 | xa2+map+xpad | 0.967 |
| ('qsa_in', 2, 5) | trial | n07 | bpre+map | 0.968 |
| ('qsa_in', 2, 8) | trial | n07 | map+bpre | 0.971 |
| ('qsa_in', 2, 16) | trial | n07 | bpre | 0.981 |
| ('qsa_in', 2, 64) | trial | n07 | xjunk+map+xpad | 0.982 |
| ('head', 2, 5) | reverted | n07 |  | 0.981 |
| ('head', 2, 8) | reverted | n07 |  | 0.979 |
| ('head', 2, 64) | reverted | n07 |  | 0.977 |
| ('head', 2, 128) | reverted | n07 |  | 0.946 |
| ('head', 1, 32) | reverted | n07 |  | 1.006 |
| ('head', 1, 128) | reverted | n07 |  | 0.948 |
| ('gdn_out', 2, 5) | trial | n01 | map+bpre | 0.954 |
| ('gdn_out', 2, 8) | trial | n01 | map+nt+bpre | 0.974 |
| ('gdn_out', 2, 16) | reverted | n01 |  | 0.967 |
| ('gdn_out', 1, 32) | reverted | n01 |  | 0.990 |
| ('gdn_out', 1, 128) | reverted | n01 |  | 0.982 |
| ('hc_down', 0, 5) | trial | n01 | nt+fg32+xa2 | 0.979 |
| ('hc_down', 0, 8) | confirmed | n01 |  | 0.980 |
| ('hc_down', 0, 8) | trial | n01 | noxu+nt+xpad | 0.989 |
| ('hc_down', 0, 16) | reverted | n01 |  | 0.989 |
| ('hc_down', 0, 32) | trial | n01 | d2+noxu | 0.965 |
| ('hc_down', 0, 128) | confirmed | n01 |  | 0.921 |
| ('hc_up', 0, 5) | trial | n01 | nt | 0.988 |
| ('hc_up', 0, 16) | reverted | n01 |  | 0.997 |
| ('hc_up', 0, 128) | confirmed | n01 |  | 0.983 |
| ('router', 0, 8) | trial | n01 | nt+map | 0.954 |
| ('router', 0, 16) | trial | n01 | nt | 0.965 |
| ('router', 0, 32) | reverted | n01 |  | 1.000 |
| ('router', 0, 128) | trial | n01 | nosb+d3+xpad | 0.979 |
| ('gdn_in', 2, 5) | trial | n01 | map+d3+bpre | 0.953 |
| ('gdn_in', 2, 16) | trial | n01 | map+d3+xsrc | 0.968 |
| ('gdn_in', 1, 5) | trial | n01 | map+xpad | 0.982 |
| ('gdn_in', 1, 16) | reverted | n01 |  | 0.975 |
| ('qsa_in', 2, 5) | reverted | n01 |  | 0.990 |
| ('qsa_in', 2, 8) | reverted | n01 |  | 0.988 |
| ('qsa_in', 2, 16) | reverted | n01 |  | 1.003 |
| ('qsa_in', 2, 64) | reverted | n01 |  | 0.988 |
| ('qsa_in', 1, 8) | trial | n01 | nosb+map | 0.970 |
| ('head', 2, 8) | trial | n01 | w4 | 0.971 |
| ('head', 1, 32) | trial | n01 | d3+bpre+map | 0.984 |
| ('gdn_out', 2, 5) | confirmed | n07 |  | 0.983 |
| ('gdn_out', 2, 8) | reverted | n07 |  | 0.989 |
| ('gdn_out', 2, 16) | trial | n07 | noxu+nt+xpad | 0.969 |
| ('gdn_out', 2, 64) | trial | n07 | map+xjunk | 0.980 |
| ('gdn_out', 1, 16) | trial | n07 | map+bpre | 0.982 |
| ('gdn_out', 1, 32) | trial | n07 | xjunk | 0.981 |
| ('hc_down', 0, 5) | reverted | n07 |  | 0.991 |
| ('hc_down', 0, 8) | reverted | n07 |  | 0.995 |
| ('hc_down', 0, 16) | trial | n07 | nt+dall | 0.989 |
| ('hc_down', 0, 32) | reverted | n07 |  | 1.009 |
| ('hc_up', 0, 5) | reverted | n07 |  | 0.997 |
| ('hc_up', 0, 16) | trial | n07 | nt | 0.986 |
| ('router', 0, 8) | reverted | n07 |  | 1.001 |
| ('router', 0, 16) | reverted | n07 |  | 0.999 |
| ('router', 0, 128) | reverted | n07 |  | 0.999 |
| ('gdn_in', 2, 5) | reverted | n07 |  | 0.990 |
| ('gdn_in', 2, 8) | trial | n07 | bpre | 0.983 |
| ('gdn_in', 2, 16) | reverted | n07 |  | 0.992 |
| ('gdn_in', 1, 5) | reverted | n07 |  | 0.995 |
| ('gdn_in', 1, 16) | trial | n07 | xa2+map+xpad | 0.962 |
| ('qsa_in', 2, 5) | trial | n07 | map | 0.964 |
| ('qsa_in', 2, 8) | trial | n07 | map | 0.958 |
| ('qsa_in', 2, 16) | trial | n07 | map | 0.965 |
| ('qsa_in', 1, 8) | reverted | n07 |  | 0.967 |
| ('head', 2, 8) | reverted | n07 |  | 0.983 |
| ('head', 1, 32) | reverted | n07 |  | 0.983 |
| ('head', 1, 128) | trial | n07 | nosb+w4+map | 0.942 |

Pending at the freeze (one-node WINs, not in the table): ('gdn_out', 2, 16) (noxu+nt+xpad 0.969 on n07), ('gdn_out', 2, 64) (map+xjunk 0.980 on n07), ('gdn_out', 1, 16) (map+bpre 0.982 on n07), ('gdn_out', 1, 32) (xjunk 0.981 on n07), ('hc_down', 0, 16) (nt+dall 0.989 on n07), ('hc_up', 0, 16) (nt 0.986 on n07), ('gdn_in', 2, 8) (bpre 0.983 on n07), ('gdn_in', 1, 16) (xa2+map+xpad 0.962 on n07), ('qsa_in', 2, 5) (map 0.964 on n07), ('qsa_in', 2, 8) (map 0.958 on n07), ('qsa_in', 2, 16) (map 0.965 on n07), ('head', 1, 128) (nosb+w4+map 0.942 on n07)

## 5. Lever ledger (climb rounds: each lever solo on the cell's pick of that round; phase ratio to the pick)

| lever | measurements | median | min | max | in WIN combinations |
|---|---|---|---|---|---|
| xpad | 220 | 0.999 | 0.941 | 1.035 | 24 |
| xjunk | 202 | 1.000 | 0.947 | 1.226 | 14 |
| fg32 | 12 | 1.001 | 0.999 | 1.023 | 2 |
| bpre | 279 | 1.001 | 0.945 | 1.049 | 29 |
| fg16 | 32 | 1.002 | 0.968 | 1.036 | 5 |
| xa2 | 118 | 1.003 | 0.943 | 1.108 | 21 |
| map | 345 | 1.004 | 0.937 | 1.411 | 51 |
| nt | 345 | 1.010 | 0.939 | 1.140 | 39 |
| d3 | 281 | 1.011 | 0.940 | 1.096 | 18 |
| d2 | 272 | 1.012 | 0.973 | 1.121 | 11 |
| d6 | 303 | 1.030 | 0.955 | 1.261 | 4 |
| w4 | 152 | 1.034 | 0.927 | 1.495 | 30 |
| nosb | 345 | 1.035 | 0.949 | 1.215 | 26 |
| dall | 205 | 1.050 | 0.955 | 1.253 | 6 |
| noxu | 345 | 1.053 | 0.971 | 1.544 | 18 |
| s/2 | 37 | 1.095 | 0.893 | 1.289 | 12 |
| xsrc | 294 | 1.111 | 0.905 | 3.308 | 5 |
| w8 | 81 | 1.133 | 1.021 | 1.324 | 0 |
| mt/2 | 188 | 1.156 | 0.896 | 1.450 | 8 |
| mt*2 | 232 | 1.162 | 0.974 | 1.538 | 0 |
| kw*2 | 127 | 1.171 | 0.940 | 1.773 | 4 |
| tb/2 | 126 | 1.214 | 0.808 | 1.782 | 3 |
| kw/2 | 193 | 1.246 | 0.924 | 1.694 | 2 |
| tb*2 | 72 | 1.300 | 1.013 | 1.669 | 0 |
| spread | 41 | 1.732 | 1.187 | 2.448 | 0 |
| bar | 345 | 1.752 | 1.090 | 2.757 | 0 |
| s*2 | 168 | 2.098 | 0.994 | 4.488 | 0 |

Lever names: `xpad` LDS row pad 32, `xjunk` branch-free staging toggle, `bpre` B operands read ahead, `fg16` / `fg32` fold loads in flight, `xa2` two activation chunks in flight, `map` rr <-> xcd, `nt` non-temporal weight loads, `d2` / `d3` / `d6` / `dall` weight prefetch depth, `w4` / `w8` waves per CTA, `nosb` no sched_barrier fences, `noxu` no cross-unit prefetch, `s/2` / `s*2` K splits, `xsrc` LDS <-> global B operands, `mt*2` / `mt/2` weight tiles per wave, `kw*2` / `kw/2` K groups per CTA, `tb/2` / `tb*2` token tiles per unit, `spread` fold=spread, `bar` the P-L3 barrier instead of persist.

## 6. Plan-of-record screens (four runs per projection group, two per node; each run's best confirmed arm vs P-L3's arm and FlyTape's plan replicas; the screen-2 lever ledger per run is in the private raw JSON)

Runs: rA1a (smci355-ccs-aus-n01-09 bus 0000:05:00); rA2a (smci355-ccs-aus-n01-09 bus 0000:05:00); rA1b (smci355-ccs-aus-n07-17 bus 0000:85:00); rA2b (smci355-ccs-aus-n07-17 bus 0000:95:00)

Per cell, phase metric unless stated (complete bf16 output on every side: fold-matched). `best` = the WP-L2a arm fastest in the cell's confirm round (per run); `base` = P-L3's verdict arm of the cell; `ft` / `ftp` = FlyTape's plan on rocKE (bf16) with P-L3's barrier / the persist barrier. Verdicts: per run, joined; a cell verdict needs every run to agree. `FT in-situ` = FlyTape's own traced phase per layer.

| cell | best (knobs, last run) | best/base phase | v | best/base stream | best/base launch | best/ft phase | v | best/ftp phase | v | best/fastest replica phase | v | ftp/ft | ftj/ft | floor/best (stream) | best phase / FT in-situ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn_out_tp1_r128 | b0+nosb+xpad: kw=4 tb=4 depth=4 bar=persist sb=False xjunk=True xpad=32 | 0.587 | WIN | 0.647 | 0.830 | 0.461 | WIN | 0.571 | WIN | 0.577 | WIN | 0.809 | 1.000 | 0.154 | - |
| gdn_out_tp1_r16 | b2+xa2: kw=8 mt=2 depth=4 bar=persist xa=2 xjunk=True | 0.525 | WIN | 0.774 | 0.844 | 0.384 | WIN | 0.551 | WIN | 0.551 | WIN | 0.693 | 1.001 | 0.294 | 0.445 |
| gdn_out_tp1_r32 | p6_bp: kw=8 mt=2 depth=4 bar=persist xjunk=True | 0.632 | WIN | 0.946 | nan | 0.451 | WIN | 0.661 | WIN | 0.666 | WIN | 0.682 | 1.000 | 0.192 | 0.553 |
| gdn_out_tp1_r5 | b0+d2+map: kw=8 xsrc=global bar=persist map=xcd | 0.504 | WIN | 0.574 | 0.661 | 0.343 | WIN | 0.479 | WIN | 0.479 | WIN | 0.716 | 1.001 | 0.406 | - |
| gdn_out_tp1_r64 | b1+xa2: waves=4 splits=6 depth=4 bar=persist xa=2 | 0.620 | WIN | 0.897 | 0.957 | 0.490 | WIN | 0.659 | WIN | 0.659 | WIN | 0.751 | 0.999 | 0.175 | 0.573 |
| gdn_out_tp1_r8 | b0+bpre+d3: kw=8 depth=3 xsrc=global bar=persist bpre=1 | 0.489 | WIN | 0.582 | 0.693 | 0.357 | WIN | 0.492 | WIN | 0.492 | WIN | 0.728 | 1.001 | 0.364 | 0.383 |
| gdn_out_tp2_r128 | b0+nosb+xpad: kw=4 tb=4 depth=4 bar=persist sb=False xjunk=True xpad=32 | 0.533 | WIN | 0.510 | 0.645 | 0.438 | WIN | 0.580 | WIN | 0.580 | WIN | 0.762 | 1.001 | 0.117 | 0.857 |
| gdn_out_tp2_r16 | b0+noxu: kw=8 depth=4 bar=persist xu=False xjunk=True | 0.431 | WIN | 0.483 | 0.566 | 0.320 | WIN | 0.482 | WIN | 0.485 | WIN | 0.665 | 0.999 | 0.309 | 0.647 |
| gdn_out_tp2_r32 | b0+nosb+noxu: kw=8 depth=4 bar=persist xu=False sb=False xjunk=True | 0.458 | WIN | 0.477 | 0.585 | 0.358 | WIN | 0.547 | WIN | 0.549 | WIN | 0.651 | 0.999 | 0.231 | 0.754 |
| gdn_out_tp2_r5 | b0+d2+nosb: waves=4 kw=4 xsrc=global bar=persist sb=False | 0.413 | WIN | 0.452 | 0.600 | 0.304 | WIN | 0.453 | WIN | 0.453 | WIN | 0.673 | 1.002 | 0.362 | - |
| gdn_out_tp2_r64 | b0+d2: kw=8 mt=2 tb=2 xsrc=global bar=persist | 0.521 | WIN | 0.495 | 0.689 | 0.437 | WIN | 0.619 | WIN | 0.619 | WIN | 0.700 | 1.000 | 0.139 | 0.937 |
| gdn_out_tp2_r8 | b0+d2+nosb: waves=4 kw=4 xsrc=global bar=persist sb=False | 0.431 | WIN | 0.464 | 0.621 | 0.317 | WIN | 0.475 | WIN | 0.477 | WIN | 0.657 | 0.999 | 0.339 | 0.688 |
| hc_down_tpx_r128 | b1+nosb: splits=10 tb=2 depth=4 bar=persist sb=False xjunk=True | 0.712 | WIN | 0.947 | 0.989 | 0.537 | WIN | 0.742 | WIN | 0.742 | WIN | 0.722 | 1.002 | 0.040 | 0.873 |
| hc_down_tpx_r16 | b1+nosb: splits=20 depth=4 bar=persist xu=False sb=False | 0.782 | WIN | 0.868 | 0.988 | 0.583 | WIN | 0.834 | WIN | 0.835 | WIN | 0.699 | 1.000 | 0.053 | 0.970 |
| hc_down_tpx_r32 | b1+nosb: waves=4 mt=2 splits=20 tb=1 depth=4 bar=persist sb=False xjunk=True | 0.702 | WIN | 0.953 | 0.981 | 0.520 | WIN | 0.728 | WIN | 0.736 | WIN | 0.714 | 0.999 | 0.052 | 0.894 |
| hc_down_tpx_r5 | b0+xa2: waves=4 splits=20 depth=4 bar=persist xa=2 | 0.676 | WIN | 0.914 | 0.936 | 0.490 | WIN | 0.717 | WIN | 0.719 | WIN | 0.686 | 0.999 | 0.095 | - |
| hc_down_tpx_r64 | b0+xpad: splits=20 tb=2 depth=4 bar=persist xa=2 xpad=32 | 0.826 | WIN | 0.991 | nan | 0.534 | WIN | 0.723 | WIN | 0.724 | WIN | 0.739 | 1.002 | 0.032 | 1.024 |
| hc_down_tpx_r8 | b0+noxu+fg16: waves=4 splits=20 depth=4 bar=persist xu=False fg=16 | 0.657 | WIN | 0.903 | 0.933 | 0.483 | WIN | 0.701 | WIN | 0.710 | WIN | 0.689 | 1.000 | 0.093 | 0.809 |
| hc_up_tpx_r128 | b0+nosb: tb=4 depth=4 bar=persist xu=False sb=False xjunk=True | 0.495 | WIN | 0.680 | 0.586 | 0.386 | WIN | 0.658 | WIN | 0.659 | WIN | 0.587 | 0.999 | 0.108 | 0.523 |
| hc_up_tpx_r16 | b0+d2+nt: waves=4 xsrc=global bar=persist xu=False wcp=nt | 0.433 | WIN | 0.520 | 0.676 | 0.320 | WIN | 0.636 | WIN | 0.642 | WIN | 0.499 | 0.999 | 0.375 | 0.548 |
| hc_up_tpx_r32 | b1+noxu: waves=4 mt=2 tb=1 depth=4 xsrc=global bar=persist xu=False | 0.488 | WIN | 0.738 | 0.681 | 0.358 | WIN | 0.731 | WIN | 0.737 | WIN | 0.490 | 1.000 | 0.240 | 0.550 |
| hc_up_tpx_r5 | b0+nt+map: waves=4 depth=4 xsrc=global bar=persist map=xcd xu=False wcp=nt | 0.441 | WIN | 0.525 | 0.674 | 0.359 | WIN | 0.667 | WIN | 0.677 | WIN | 0.551 | 0.997 | 0.378 | - |
| hc_up_tpx_r64 | b1+noxu: waves=4 mt=2 tb=2 depth=4 xsrc=global bar=persist xu=False | 0.479 | WIN | 0.691 | 0.656 | 0.390 | WIN | 0.715 | WIN | 0.715 | WIN | 0.541 | 0.997 | 0.171 | 0.526 |
| hc_up_tpx_r8 | b2+noxu: waves=4 depth=4 xsrc=global bar=persist xu=False wcp=nt | 0.482 | WIN | 0.520 | 0.672 | 0.366 | WIN | 0.700 | WIN | 0.708 | WIN | 0.518 | 1.000 | 0.382 | 0.624 |
| router_tpx_r128 | b0+xa2+noxu: waves=4 kw=4 tb=2 depth=4 bar=persist xu=False xa=2 xjunk=True | 0.427 | WIN | 0.495 | 0.580 | 0.265 | WIN | 0.323 | WIN | 0.323 | WIN | 0.820 | 0.999 | 0.049 | 0.249 |
| router_tpx_r16 | b0+nosb+nt: waves=4 kw=4 depth=4 bar=persist xu=False wcp=nt sb=False xjunk=True | 0.394 | WIN | 0.489 | 0.640 | 0.339 | WIN | 0.508 | WIN | 0.509 | WIN | 0.667 | 0.999 | 0.069 | 0.219 |
| router_tpx_r32 | b0+nosb+xpad: waves=4 kw=4 tb=1 depth=4 bar=persist xu=False sb=False xjunk=True xpad=32 | 0.358 | WIN | 0.485 | 0.644 | 0.303 | WIN | 0.425 | WIN | 0.431 | WIN | 0.713 | 1.002 | 0.070 | 0.218 |
| router_tpx_r5 | b0+nt: waves=4 kw=4 depth=4 xsrc=global bar=persist xu=False wcp=nt | 0.405 | WIN | 0.692 | 0.688 | 0.343 | WIN | 0.522 | WIN | 0.522 | WIN | 0.660 | 1.000 | 0.071 | - |
| router_tpx_r64 | b2+nosb: waves=4 kw=4 tb=1 xsrc=global bar=persist sb=False | 0.398 | WIN | 0.487 | 0.705 | 0.297 | WIN | 0.389 | WIN | 0.392 | WIN | 0.761 | 0.999 | 0.066 | 0.229 |
| router_tpx_r8 | b2+nosb: waves=4 kw=4 xsrc=global bar=persist sb=False | 0.407 | WIN | 0.622 | 0.720 | 0.345 | WIN | 0.517 | WIN | 0.519 | WIN | 0.666 | 1.000 | 0.078 | 0.224 |

Verdict tallies (cells): best vs base: WIN 30; best vs ft: WIN 30; best vs ftp: WIN 30; best vs fastest replica: WIN 30


Runs: rB1a (smci355-ccs-aus-n01-09 bus 0000:05:00); rB2a (smci355-ccs-aus-n01-09 bus 0000:05:00); rB1b (smci355-ccs-aus-n07-17 bus 0000:95:00); rB2b (smci355-ccs-aus-n07-17 bus 0000:65:00)

Per cell, phase metric unless stated (complete bf16 output on every side: fold-matched). `best` = the WP-L2a arm fastest in the cell's confirm round (per run); `base` = P-L3's verdict arm of the cell; `ft` / `ftp` = FlyTape's plan on rocKE (bf16) with P-L3's barrier / the persist barrier. Verdicts: per run, joined; a cell verdict needs every run to agree. `FT in-situ` = FlyTape's own traced phase per layer.

| cell | best (knobs, last run) | best/base phase | v | best/base stream | best/base launch | best/ft phase | v | best/ftp phase | v | best/fastest replica phase | v | ftp/ft | ftj/ft | floor/best (stream) | best phase / FT in-situ |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1_r128 | b1+nosb: mt=2 tb=4 depth=4 bar=persist sb=False xjunk=True | 0.708 | WIN | 0.945 | 0.940 | 0.252 | WIN | 0.285 | WIN | 0.285 | WIN | 0.889 | 0.998 | 0.269 | - |
| gdn_in_tp1_r16 | b0+xa2+nosb: kw=4 mt=4 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.571 | WIN | 0.744 | 0.938 | 0.206 | WIN | 0.251 | WIN | 0.251 | WIN | 0.833 | 1.001 | 0.560 | 0.239 |
| gdn_in_tp1_r32 | b2+xa2: kw=2 mt=2 depth=4 bar=persist xa=2 xjunk=True | 0.594 | WIN | 0.899 | nan | 0.227 | WIN | 0.270 | WIN | 0.270 | WIN | 0.841 | 0.998 | 0.448 | 0.250 |
| gdn_in_tp1_r5 | b0+nosb+xjunk: kw=4 mt=4 depth=4 bar=persist wcp=nt sb=False | 0.569 | WIN | 0.704 | 0.921 | 0.206 | WIN | 0.247 | WIN | 0.247 | WIN | 0.835 | 1.005 | 0.607 | - |
| gdn_in_tp1_r64 | b0+nt+xpad: depth=4 bar=persist wcp=nt xa=2 xjunk=True xpad=32 | 0.689 | WIN | 0.954 | 0.940 | 0.235 | WIN | 0.274 | WIN | 0.274 | WIN | 0.859 | 1.000 | 0.391 | 0.234 |
| gdn_in_tp1_r8 | b1+nt: kw=4 mt=4 depth=4 bar=persist wcp=nt sb=False xjunk=True | 0.560 | WIN | 0.707 | 0.926 | 0.205 | WIN | 0.245 | WIN | 0.245 | WIN | 0.841 | 1.005 | 0.595 | 0.235 |
| gdn_in_tp2_r128 | b1+nosb: tb=4 depth=4 bar=persist sb=False | 0.609 | WIN | 0.852 | 0.911 | 0.257 | WIN | 0.297 | WIN | 0.300 | WIN | 0.864 | 1.007 | 0.195 | 0.362 |
| gdn_in_tp2_r16 | p0_bp+d2: waves=4 kw=4 mt=4 xsrc=global bar=persist | 0.426 | WIN | 0.588 | 0.763 | 0.209 | WIN | 0.252 | WIN | 0.255 | WIN | 0.830 | 1.007 | 0.458 | 0.361 |
| gdn_in_tp2_r32 | b0+nosb+nt: kw=4 mt=2 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.474 | WIN | 0.700 | 0.940 | 0.224 | WIN | 0.288 | WIN | 0.295 | WIN | 0.778 | 1.007 | 0.362 | 0.390 |
| gdn_in_tp2_r5 | b0+nosb+nt: waves=4 kw=4 mt=4 xsrc=global bar=persist wcp=nt sb=False | 0.396 | WIN | 0.585 | 0.544 | 0.216 | WIN | 0.246 | WIN | 0.250 | WIN | 0.882 | 1.008 | 0.536 | - |
| gdn_in_tp2_r64 | b0+nosb+xpad: kw=2 depth=4 bar=persist xa=2 sb=False xjunk=True xpad=32 | 0.592 | WIN | 0.932 | nan | 0.282 | WIN | 0.339 | WIN | 0.346 | WIN | 0.827 | 1.005 | 0.239 | 0.426 |
| gdn_in_tp2_r8 | b1+nt: waves=4 kw=4 mt=4 xsrc=global bar=persist wcp=nt | 0.407 | WIN | 0.593 | 0.556 | 0.215 | WIN | 0.254 | WIN | 0.256 | WIN | 0.849 | 1.008 | 0.506 | 0.354 |
| head_tp1_r128 | b0+bpre+nosb: waves=4 mt=4 depth=4 bar=persist bpre=2 sb=False xjunk=True | 0.900 | WIN | 0.989 | 0.990 | 0.300 | WIN | 0.300 | WIN | 0.300 | WIN | 0.997 | 1.010 | 0.517 | - |
| head_tp1_r16 | p1_bp+nt: mt=4 depth=4 bar=persist wcp=nt xjunk=True | 0.699 | WIN | 0.731 | 0.726 | 0.235 | WIN | 0.241 | WIN | 0.241 | WIN | 0.975 | 1.000 | 0.928 | 0.260 |
| head_tp1_r32 | b1+nt: mt=4 depth=4 bar=persist map=xcd wcp=nt xjunk=True | 0.842 | WIN | 0.930 | 0.908 | 0.251 | WIN | 0.257 | WIN | 0.257 | WIN | 0.978 | 1.001 | 0.836 | 0.273 |
| head_tp1_r5 | b2+nt: mt=4 depth=3 xsrc=global bar=persist wcp=nt | 0.637 | WIN | 0.667 | 0.655 | 0.211 | WIN | 0.216 | WIN | 0.216 | WIN | 0.979 | 1.000 | 1.044 | - |
| head_tp1_r64 | b0+d3+xpad: mt=4 depth=3 bar=persist wcp=nt xjunk=True xpad=32 | 0.832 | WIN | 0.915 | 0.891 | 0.267 | WIN | 0.274 | WIN | 0.274 | WIN | 0.972 | 0.999 | 0.752 | 0.327 |
| head_tp1_r8 | b0+noxu+d2: mt=4 xsrc=global bar=persist xu=False wcp=nt | 0.656 | WIN | 0.687 | 0.689 | 0.213 | WIN | 0.218 | WIN | 0.218 | WIN | 0.977 | 1.000 | 1.033 | 0.240 |
| head_tp2_r128 | b0+d3+xpad: mt=2 depth=3 bar=persist wcp=nt xjunk=True xpad=32 | 0.820 | WIN | 0.992 | 0.990 | 0.293 | WIN | 0.305 | WIN | 0.305 | WIN | 0.962 | 1.008 | 0.514 | 0.373 |
| head_tp2_r16 | p0_bp+nt: mt=4 depth=4 xsrc=global bar=persist wcp=nt | 0.659 | WIN | 0.736 | 0.759 | 0.233 | WIN | 0.247 | WIN | 0.247 | WIN | 0.943 | 0.999 | 0.910 | 0.248 |
| head_tp2_r32 | b0+d3+nosb: mt=4 depth=3 bar=persist wcp=nt sb=False xjunk=True | 0.734 | WIN | 0.922 | 0.869 | 0.250 | WIN | 0.265 | WIN | 0.266 | WIN | 0.939 | 1.001 | 0.837 | 0.263 |
| head_tp2_r5 | b0+d3: mt=4 depth=3 xsrc=global bar=persist wcp=nt | 0.609 | WIN | 0.659 | 0.660 | 0.213 | WIN | 0.226 | WIN | 0.226 | WIN | 0.949 | 0.999 | 1.032 | - |
| head_tp2_r64 | b0+d3+bpre: mt=4 depth=3 bar=persist wcp=nt bpre=2 xjunk=True | 0.738 | WIN | 0.905 | 0.917 | 0.251 | WIN | 0.267 | WIN | 0.267 | WIN | 0.937 | 0.999 | 0.801 | 0.286 |
| head_tp2_r8 | p0_bp+nt: mt=4 depth=4 xsrc=global bar=persist wcp=nt | 0.631 | WIN | 0.696 | 0.691 | 0.227 | WIN | 0.239 | WIN | 0.239 | WIN | 0.947 | 0.999 | 0.957 | 0.245 |
| qsa_in_tp1_r128 | b0+nosb+map: tb=4 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.622 | WIN | 0.934 | 0.893 | 0.223 | WIN | 0.280 | WIN | 0.282 | WIN | 0.795 | 1.003 | 0.335 | - |
| qsa_in_tp1_r16 | b0+nosb+d3: waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.457 | WIN | 0.533 | 0.681 | 0.189 | WIN | 0.261 | WIN | 0.262 | WIN | 0.727 | 1.001 | 0.669 | 0.263 |
| qsa_in_tp1_r32 | b0+nosb+xa2: kw=4 mt=2 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.470 | WIN | 0.621 | 0.833 | 0.199 | WIN | 0.281 | WIN | 0.282 | WIN | 0.708 | 1.002 | 0.556 | 0.282 |
| qsa_in_tp1_r5 | b0+nosb+d3: waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.442 | WIN | 0.440 | 0.632 | 0.186 | WIN | 0.251 | WIN | 0.253 | WIN | 0.739 | 0.997 | 0.822 | - |
| qsa_in_tp1_r64 | b0+d3: kw=4 mt=2 depth=3 bar=persist sb=False xjunk=True | 0.637 | WIN | 0.870 | 0.927 | 0.244 | WIN | 0.331 | WIN | 0.332 | WIN | 0.738 | 0.998 | 0.369 | 0.305 |
| qsa_in_tp1_r8 | b0+nosb+d3: waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.442 | WIN | 0.477 | 0.677 | 0.185 | WIN | 0.253 | WIN | 0.254 | WIN | 0.735 | 0.999 | 0.755 | 0.255 |
| qsa_in_tp2_r128 | b0+xa2: kw=2 tb=4 depth=4 bar=persist xa=2 xjunk=True | 0.549 | WIN | 0.708 | 0.815 | 0.301 | WIN | 0.406 | WIN | 0.407 | WIN | 0.747 | 1.009 | 0.207 | 0.430 |
| qsa_in_tp2_r16 | b0+d2+nosb: waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.357 | WIN | 0.430 | 0.484 | 0.207 | WIN | 0.306 | WIN | 0.306 | WIN | 0.668 | 0.999 | 0.564 | 0.340 |
| qsa_in_tp2_r32 | b1+nosb: waves=4 kw=4 mt=2 depth=4 bar=persist sb=False xjunk=True | 0.426 | WIN | 0.501 | 0.712 | 0.244 | WIN | 0.364 | WIN | 0.366 | WIN | 0.677 | 1.003 | 0.446 | 0.391 |
| qsa_in_tp2_r5 | b1+nosb: waves=4 kw=4 mt=2 xsrc=global bar=persist sb=False | 0.359 | WIN | 0.431 | 0.456 | 0.218 | WIN | 0.295 | WIN | 0.297 | WIN | 0.741 | 1.002 | 0.670 | - |
| qsa_in_tp2_r64 | b0+nosb: waves=4 kw=4 mt=2 depth=4 bar=persist sb=False xjunk=True | 0.480 | WIN | 0.504 | 0.548 | 0.302 | WIN | 0.415 | WIN | 0.415 | WIN | 0.729 | 0.998 | 0.290 | 0.433 |
| qsa_in_tp2_r8 | b0+nt+nosb: waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.366 | WIN | 0.440 | 0.475 | 0.218 | WIN | 0.296 | WIN | 0.296 | WIN | 0.739 | 1.000 | 0.632 | 0.320 |

Verdict tallies (cells): best vs base: WIN 36; best vs ft: WIN 36; best vs ftp: WIN 36; best vs fastest replica: WIN 36

