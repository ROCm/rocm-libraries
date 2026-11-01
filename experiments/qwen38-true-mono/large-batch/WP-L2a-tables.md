# WP-L2a tables (ratios)

Generated from the runs' JSON by `hackathon-prep/qwen38/truemono-lb/wp-l2a/{picks_report,climb_report,l2a_report}.py`;
absolute numbers in `hackathon-prep/private-perf/qwen38-true-mono/large-batch/WP-L2a.md`. See `WP-L2a.md` for method and
caveats. The tables of the round-6 plan table (before the 16-round climb) are kept in `WP-L2a-tables-r6.md`.

## 1. Final picks (frozen after climb round 16): four confirm runs (venv, two per node)

Runs: rF1a (smci355-ccs-aus-n01-09 bus 0000:05:00); rF2a (smci355-ccs-aus-n01-09 bus 0000:15:00); rF1b (smci355-ccs-aus-n07-17 bus 0000:75:00); rF2b (smci355-ccs-aus-n07-17 bus 0000:95:00)

The plan table's pick (`e1_plans.PLANS`) per cell, confirm rounds only (24 shuffled rounds, numerics first, second-graph and recompiled controls on every side). Verdicts per run joined (a cell verdict needs every run to agree; else MIXED). `fastest replica` = the fastest of FlyTape's plan on rocKE with P-L3's barrier / the persist barrier, with predicated / branch-free staging. FT in-situ: FlyTape's own traced phase per layer; for out-proj the stricter of gdn_out / qsa_out. Phase metric unless stated.

| cell | pick | pick/base phase | v | pick/base stream | pick/base launch | pick/fastest replica phase | v | pick/ft phase | floor/pick (stream) | pick phase / FT in-situ (same job) | pick phase / FT in-situ (W4 trace) | pick/round-6 pick phase | v |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1_r128 | waves=4 mt=4 tb=4 depth=4 bar=persist map=xcd wcp=nt xa=2 bpre=2 sb=False xpad=32 | 0.663 | WIN | 0.883 | 0.880 | 0.266 | WIN | 0.233 | 0.284 | - | - | 0.957 | WIN |
| gdn_in_tp1_r16 | kw=4 mt=4 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.552 | WIN | 0.730 | 0.949 | 0.238 | WIN | 0.196 | 0.572 | - | 0.230 | 0.965 | MIXED(WIN/TIE/TIE/TIE) |
| gdn_in_tp1_r32 | waves=4 mt=2 depth=4 bar=persist wcp=nt xa=2 bpre=2 sb=False xjunk=True | 0.527 | WIN | 0.708 | 0.792 | 0.240 | WIN | 0.202 | 0.568 | - | 0.223 | 0.891 | WIN |
| gdn_in_tp1_r5 | waves=4 mt=2 depth=4 bar=persist wcp=nt xa=2 bpre=1 sb=False xjunk=True | 0.515 | WIN | 0.593 | 0.737 | 0.223 | WIN | 0.186 | 0.711 | - | - | 0.917 | WIN |
| gdn_in_tp1_r64 | waves=4 mt=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xpad=32 | 0.652 | WIN | 0.861 | 0.834 | 0.265 | WIN | 0.223 | 0.432 | - | 0.223 | - | same |
| gdn_in_tp1_r8 | kw=4 mt=4 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.559 | WIN | 0.698 | 0.937 | 0.248 | WIN | 0.201 | 0.605 | - | 0.231 | - | same |
| gdn_in_tp2_r128 | waves=4 mt=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.548 | WIN | 0.715 | 0.762 | 0.267 | WIN | 0.223 | 0.231 | - | 0.319 | - | same |
| gdn_in_tp2_r16 | waves=4 kw=4 mt=4 xsrc=global bar=persist wcp=nt sb=False | 0.423 | WIN | 0.583 | 0.760 | 0.270 | WIN | 0.204 | 0.461 | - | 0.359 | - | same |
| gdn_in_tp2_r32 | waves=4 kw=2 mt=2 depth=4 bar=persist xa=2 bpre=2 sb=False xjunk=True | 0.459 | WIN | 0.603 | 0.788 | 0.281 | WIN | 0.217 | 0.420 | - | 0.377 | 0.965 | MIXED(WIN/TIE/WIN/WIN) |
| gdn_in_tp2_r5 | waves=4 kw=4 mt=4 depth=4 xsrc=global bar=persist map=xcd wcp=nt sb=False | 0.394 | WIN | 0.571 | 0.562 | 0.259 | WIN | 0.207 | 0.530 | - | - | - | same |
| gdn_in_tp2_r64 | waves=4 mt=2 tb=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xjunk=True | 0.497 | WIN | 0.700 | 0.796 | 0.292 | WIN | 0.237 | 0.317 | - | 0.361 | 0.833 | WIN |
| gdn_in_tp2_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.398 | WIN | 0.565 | 0.560 | 0.258 | WIN | 0.205 | 0.515 | - | 0.349 | - | same |
| gdn_out_tp1_r128 | kw=4 mt=2 tb=2 depth=4 bar=persist map=xcd xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.507 | WIN | 0.556 | 0.758 | 0.487 | WIN | 0.391 | 0.183 | - | - | 0.861 | WIN |
| gdn_out_tp1_r16 | waves=4 kw=4 bar=persist wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.437 | WIN | 0.505 | 0.651 | 0.453 | WIN | 0.317 | 0.448 | - | 0.389 | 0.940 | WIN |
| gdn_out_tp1_r32 | waves=4 kw=4 mt=2 tb=1 depth=4 bar=persist xa=2 bpre=1 sb=False xjunk=True xpad=32 | 0.450 | WIN | 0.520 | 0.679 | 0.469 | WIN | 0.315 | 0.346 | - | 0.434 | - | same |
| gdn_out_tp1_r5 | waves=4 kw=4 depth=3 xsrc=global bar=persist map=xcd sb=False | 0.455 | WIN | 0.493 | 0.602 | 0.439 | WIN | 0.317 | 0.476 | - | - | - | same |
| gdn_out_tp1_r64 | waves=4 splits=6 depth=4 bar=persist xa=2 sb=False | 0.582 | WIN | 0.876 | 0.933 | 0.628 | WIN | 0.474 | 0.183 | - | 0.568 | - | same |
| gdn_out_tp1_r8 | waves=4 kw=4 depth=3 xsrc=global bar=persist map=xcd sb=False | 0.450 | WIN | 0.494 | 0.630 | 0.459 | WIN | 0.323 | 0.434 | - | 0.379 | - | same |
| gdn_out_tp2_r128 | waves=4 kw=2 mt=2 tb=2 depth=6 bar=persist map=xcd xa=2 sb=False xjunk=True xpad=32 | 0.391 | WIN | 0.346 | 0.578 | 0.428 | WIN | 0.323 | 0.175 | - | 0.631 | 0.801 | WIN |
| gdn_out_tp2_r16 | kw=8 depth=4 bar=persist map=xcd xu=False wcp=nt sb=False xjunk=True xpad=32 | 0.423 | WIN | 0.460 | 0.525 | 0.473 | WIN | 0.308 | 0.330 | - | 0.823 | 0.968 | MIXED(TIE/WIN/WIN/WIN) |
| gdn_out_tp2_r32 | kw=4 tb=1 bar=persist xa=2 xjunk=True | 0.416 | WIN | 0.415 | 0.707 | 0.493 | WIN | 0.326 | 0.259 | - | 0.843 | 0.942 | WIN |
| gdn_out_tp2_r5 | waves=4 kw=4 xsrc=global bar=persist map=xcd bpre=1 sb=False | 0.409 | WIN | 0.441 | 0.571 | 0.453 | WIN | 0.297 | 0.371 | - | - | - | same |
| gdn_out_tp2_r64 | waves=4 kw=2 tb=2 depth=3 bar=persist map=xcd xu=False xa=2 bpre=2 sb=False xjunk=True | 0.418 | WIN | 0.340 | 0.512 | 0.497 | WIN | 0.343 | 0.199 | - | 0.747 | 0.943 | WIN |
| gdn_out_tp2_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.423 | WIN | 0.466 | 0.618 | 0.474 | WIN | 0.308 | 0.342 | - | 0.839 | - | same |
| hc_down_tpx_r128 | waves=4 splits=5 tb=2 depth=3 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.594 | WIN | 0.696 | 0.767 | 0.625 | WIN | 0.454 | 0.053 | - | 0.727 | - | same |
| hc_down_tpx_r16 | waves=4 splits=10 depth=4 bar=persist xu=False wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.601 | WIN | 0.585 | 0.797 | 0.660 | WIN | 0.446 | 0.083 | - | 0.756 | - | same |
| hc_down_tpx_r32 | waves=4 splits=10 tb=1 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.596 | WIN | 0.596 | 0.821 | 0.607 | WIN | 0.438 | 0.082 | - | 0.767 | - | same |
| hc_down_tpx_r5 | waves=4 kw=2 mt=2 splits=10 xsrc=global bar=persist wcp=nt bpre=1 sb=False | 0.575 | WIN | 0.866 | 0.902 | 0.623 | WIN | 0.423 | 0.100 | - | - | 0.863 | WIN |
| hc_down_tpx_r64 | waves=4 kw=2 mt=2 splits=5 tb=1 bar=persist xa=2 bpre=1 fg=16 sb=False xpad=32 | 0.540 | WIN | 0.415 | 0.693 | 0.472 | WIN | 0.352 | 0.077 | - | 0.676 | 0.927 | WIN |
| hc_down_tpx_r8 | waves=4 splits=10 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.604 | WIN | 0.927 | 0.959 | 0.675 | WIN | 0.456 | 0.090 | - | 0.756 | - | same |
| hc_up_tpx_r128 | tb=4 depth=4 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.501 | WIN | 0.663 | 0.571 | 0.679 | WIN | 0.382 | 0.109 | - | 0.524 | - | same |
| hc_up_tpx_r16 | waves=4 xsrc=global bar=persist xu=False bpre=1 | 0.448 | WIN | 0.521 | 0.676 | 0.666 | WIN | 0.336 | 0.376 | - | 0.577 | - | same |
| hc_up_tpx_r32 | waves=4 depth=6 bar=persist xu=False wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.497 | WIN | 0.684 | 0.644 | 0.724 | WIN | 0.352 | 0.261 | - | 0.554 | 0.967 | MIXED(WIN/WIN/WIN/TIE) |
| hc_up_tpx_r5 | waves=4 depth=3 xsrc=global bar=persist xu=False bpre=1 | 0.439 | WIN | 0.526 | 0.672 | 0.667 | WIN | 0.357 | 0.375 | - | - | - | same |
| hc_up_tpx_r64 | waves=4 mt=2 tb=2 depth=4 xsrc=global bar=persist xu=False | 0.479 | WIN | 0.688 | 0.651 | 0.718 | WIN | 0.389 | 0.173 | - | 0.522 | - | same |
| hc_up_tpx_r8 | waves=4 xsrc=global bar=persist xu=False wcp=nt | 0.451 | WIN | 0.519 | 0.672 | 0.673 | WIN | 0.349 | 0.379 | - | 0.594 | - | same |
| head_tp1_r128 | waves=4 mt=4 depth=3 bar=persist map=xcd bpre=2 sb=False xjunk=True xpad=32 | 0.860 | WIN | 0.934 | 0.936 | 0.286 | WIN | 0.285 | 0.543 | - | - | 0.948 | MIXED(WIN/TIE/TIE/TIE) |
| head_tp1_r16 | waves=4 mt=4 depth=4 bar=persist map=xcd wcp=nt sb=False xjunk=True | 0.685 | WIN | 0.738 | 0.692 | 0.242 | WIN | 0.236 | 0.880 | - | 0.263 | 0.975 | TIE |
| head_tp1_r32 | mt=4 depth=4 bar=persist wcp=nt xa=2 xjunk=True xpad=32 | 0.846 | WIN | 0.936 | 0.906 | 0.264 | WIN | 0.256 | 0.827 | - | 0.279 | - | same |
| head_tp1_r5 | mt=4 depth=4 xsrc=global bar=persist xu=False wcp=nt | 0.631 | WIN | 0.665 | 0.635 | 0.218 | WIN | 0.213 | 1.015 | - | - | - | same |
| head_tp1_r64 | mt=4 depth=4 bar=persist map=xcd wcp=nt xjunk=True xpad=32 | 0.837 | WIN | 0.927 | 0.883 | 0.273 | WIN | 0.267 | 0.744 | - | 0.330 | - | same |
| head_tp1_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt | 0.661 | WIN | 0.702 | 0.675 | 0.233 | WIN | 0.229 | 0.943 | - | 0.256 | - | same |
| head_tp2_r128 | waves=4 mt=4 bar=persist wcp=nt bpre=2 sb=False xpad=32 | 0.776 | WIN | 0.928 | 0.907 | 0.289 | WIN | 0.280 | 0.550 | - | 0.354 | 0.952 | WIN |
| head_tp2_r16 | waves=4 mt=4 depth=4 bar=persist map=xcd wcp=nt xa=2 xjunk=True | 0.626 | WIN | 0.713 | 0.707 | 0.238 | WIN | 0.226 | 0.944 | - | 0.238 | 0.951 | MIXED(TIE/WIN/WIN/TIE) |
| head_tp2_r32 | mt=4 depth=3 bar=persist wcp=nt sb=False xjunk=True | 0.731 | WIN | 0.923 | 0.897 | 0.260 | WIN | 0.242 | 0.890 | - | 0.251 | - | same |
| head_tp2_r5 | waves=4 mt=2 depth=3 xsrc=global bar=persist xu=False wcp=nt bpre=1 | 0.588 | WIN | 0.656 | 0.656 | 0.220 | WIN | 0.210 | 1.032 | - | - | 0.975 | TIE |
| head_tp2_r64 | mt=4 depth=3 bar=persist wcp=nt xjunk=True xpad=32 | 0.733 | WIN | 0.910 | 0.883 | 0.283 | WIN | 0.265 | 0.752 | - | 0.301 | - | same |
| head_tp2_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt bpre=1 sb=False | 0.625 | WIN | 0.688 | 0.715 | 0.234 | WIN | 0.221 | 0.988 | - | 0.237 | - | same |
| qsa_in_tp1_r128 | waves=4 mt=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False | 0.593 | WIN | 0.844 | 0.840 | 0.266 | WIN | 0.212 | 0.367 | - | - | - | same |
| qsa_in_tp1_r16 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.440 | WIN | 0.502 | 0.716 | 0.250 | WIN | 0.182 | 0.712 | - | 0.252 | - | same |
| qsa_in_tp1_r32 | kw=4 mt=2 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.477 | WIN | 0.622 | 0.833 | 0.286 | WIN | 0.206 | 0.555 | - | 0.286 | - | same |
| qsa_in_tp1_r5 | waves=4 kw=4 mt=4 depth=4 xsrc=global bar=persist wcp=nt sb=False | 0.435 | WIN | 0.438 | 0.663 | 0.248 | WIN | 0.182 | 0.822 | - | - | - | same |
| qsa_in_tp1_r64 | kw=2 mt=2 tb=2 bar=persist map=xcd wcp=nt sb=False | 0.620 | WIN | 0.809 | 0.833 | 0.323 | WIN | 0.241 | 0.396 | - | 0.296 | 0.964 | WIN |
| qsa_in_tp1_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt | 0.441 | WIN | 0.466 | 0.677 | 0.254 | WIN | 0.184 | 0.772 | - | 0.254 | - | same |
| qsa_in_tp2_r128 | waves=4 mt=2 tb=2 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.442 | WIN | 0.507 | 0.595 | 0.325 | WIN | 0.235 | 0.287 | - | 0.343 | 0.826 | WIN |
| qsa_in_tp2_r16 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.349 | WIN | 0.417 | 0.486 | 0.298 | WIN | 0.208 | 0.570 | - | 0.330 | - | same |
| qsa_in_tp2_r32 | waves=4 kw=2 mt=2 tb=1 depth=6 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.394 | WIN | 0.455 | 0.714 | 0.342 | WIN | 0.234 | 0.489 | - | 0.361 | 0.957 | WIN |
| qsa_in_tp2_r5 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.352 | WIN | 0.378 | 0.462 | 0.288 | WIN | 0.213 | 0.704 | - | - | - | same |
| qsa_in_tp2_r64 | waves=4 kw=2 mt=2 tb=2 depth=6 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.402 | WIN | 0.393 | 0.562 | 0.347 | WIN | 0.252 | 0.369 | - | 0.360 | 0.832 | WIN |
| qsa_in_tp2_r8 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.367 | WIN | 0.410 | 0.475 | 0.297 | WIN | 0.216 | 0.656 | - | 0.320 | - | same |
| router_tpx_r128 | waves=4 kw=4 tb=2 depth=4 bar=persist xu=False xa=2 xjunk=True | 0.420 | WIN | 0.500 | 0.574 | 0.321 | WIN | 0.261 | 0.050 | - | 0.244 | - | same |
| router_tpx_r16 | waves=4 kw=4 depth=4 bar=persist xu=False sb=False xjunk=True xpad=32 | 0.389 | WIN | 0.490 | 0.655 | 0.508 | WIN | 0.337 | 0.069 | - | 0.216 | - | same |
| router_tpx_r32 | waves=4 kw=4 tb=1 depth=4 bar=persist xu=False sb=False xjunk=True | 0.359 | WIN | 0.484 | 0.645 | 0.438 | WIN | 0.307 | 0.070 | - | 0.222 | - | same |
| router_tpx_r5 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.387 | WIN | 0.583 | 0.751 | 0.519 | WIN | 0.334 | 0.084 | - | - | - | same |
| router_tpx_r64 | waves=4 kw=4 tb=1 bar=persist xa=2 sb=False xjunk=True | 0.362 | WIN | 0.386 | 0.670 | 0.366 | WIN | 0.270 | 0.083 | - | 0.212 | 0.924 | WIN |
| router_tpx_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.397 | WIN | 0.624 | 0.787 | 0.529 | WIN | 0.334 | 0.078 | - | 0.222 | - | same |

Verdict tallies (cells): pick vs base: WIN 66; pick vs fastest replica: WIN 66; pick vs round-6 pick: MIXED 6, TIE 2, WIN 17

### Per projection (ranges over rows 5-128 of the per-cell medians)

| projection | pick/base phase | pick/base stream | pick/base launch | pick/fastest replica phase | pick/ft phase | floor/pick (stream) | pick / FT in-situ (same job) | pick / FT in-situ (W4 trace) | pick/round-6 pick (changed cells) |
|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1 | 0.52-0.66 | 0.59-0.88 | 0.74-0.95 | 0.22-0.27 | 0.19-0.23 | 0.28-0.71 | - | 0.22-0.23 | 0.89-0.96 |
| gdn_in_tp2 | 0.39-0.55 | 0.56-0.71 | 0.56-0.80 | 0.26-0.29 | 0.20-0.24 | 0.23-0.53 | - | 0.32-0.38 | 0.83-0.96 |
| gdn_out_tp1 | 0.44-0.58 | 0.49-0.88 | 0.60-0.93 | 0.44-0.63 | 0.32-0.47 | 0.18-0.48 | - | 0.38-0.57 | 0.86-0.94 |
| gdn_out_tp2 | 0.39-0.42 | 0.34-0.47 | 0.51-0.71 | 0.43-0.50 | 0.30-0.34 | 0.17-0.37 | - | 0.63-0.84 | 0.80-0.97 |
| hc_down_tpx | 0.54-0.60 | 0.41-0.93 | 0.69-0.96 | 0.47-0.68 | 0.35-0.46 | 0.05-0.10 | - | 0.68-0.77 | 0.86-0.93 |
| hc_up_tpx | 0.44-0.50 | 0.52-0.69 | 0.57-0.68 | 0.67-0.72 | 0.34-0.39 | 0.11-0.38 | - | 0.52-0.59 | 0.97-0.97 |
| head_tp1 | 0.63-0.86 | 0.67-0.94 | 0.64-0.94 | 0.22-0.29 | 0.21-0.28 | 0.54-1.01 | - | 0.26-0.33 | 0.95-0.97 |
| head_tp2 | 0.59-0.78 | 0.66-0.93 | 0.66-0.91 | 0.22-0.29 | 0.21-0.28 | 0.55-1.03 | - | 0.24-0.35 | 0.95-0.97 |
| qsa_in_tp1 | 0.43-0.62 | 0.44-0.84 | 0.66-0.84 | 0.25-0.32 | 0.18-0.24 | 0.37-0.82 | - | 0.25-0.30 | 0.96-0.96 |
| qsa_in_tp2 | 0.35-0.44 | 0.38-0.51 | 0.46-0.71 | 0.29-0.35 | 0.21-0.25 | 0.29-0.70 | - | 0.32-0.36 | 0.83-0.96 |
| router_tpx | 0.36-0.42 | 0.39-0.62 | 0.57-0.79 | 0.32-0.53 | 0.26-0.34 | 0.05-0.08 | - | 0.21-0.24 | 0.92-0.92 |

Numerics of these runs: per-cell checks 3368 (failing 0); suite records 992 (mutants caught 752 / 752); chain checks 192; barrier timeouts 0.

## 2. Final picks: the container run with FlyTape traced in the same job (before and after the LB timing; the ROCm 10 image's COMGR)

Runs: C3 (smci355-ccs-aus-n01-09 bus 0000:05:00)

The plan table's pick (`e1_plans.PLANS`) per cell, confirm rounds only (24 shuffled rounds, numerics first, second-graph and recompiled controls on every side). Verdicts per run joined (a cell verdict needs every run to agree; else MIXED). `fastest replica` = the fastest of FlyTape's plan on rocKE with P-L3's barrier / the persist barrier, with predicated / branch-free staging. FT in-situ: FlyTape's own traced phase per layer; for out-proj the stricter of gdn_out / qsa_out. Phase metric unless stated.

| cell | pick | pick/base phase | v | pick/base stream | pick/base launch | pick/fastest replica phase | v | pick/ft phase | floor/pick (stream) | pick phase / FT in-situ (same job) | pick phase / FT in-situ (W4 trace) | pick/round-6 pick phase | v |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1_r16 | kw=4 mt=4 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.565 | WIN | 0.837 | 1.061 | 0.259 | WIN | 0.205 | 0.594 | 0.230 | - | 0.982 | TIE |
| gdn_in_tp1_r32 | waves=4 mt=2 depth=4 bar=persist wcp=nt xa=2 bpre=2 sb=False xjunk=True | 0.540 | WIN | 0.727 | 0.804 | 0.248 | WIN | 0.204 | 0.584 | 0.228 | - | 0.897 | WIN |
| gdn_in_tp1_r64 | waves=4 mt=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xpad=32 | 0.666 | WIN | 0.880 | 0.861 | 0.269 | WIN | 0.226 | 0.441 | 0.227 | - | - | same |
| gdn_in_tp1_r8 | kw=4 mt=4 depth=4 bar=persist wcp=nt xa=2 sb=False xjunk=True | 0.569 | WIN | 0.803 | 1.055 | 0.259 | WIN | 0.208 | 0.624 | 0.229 | - | - | same |
| gdn_in_tp2_r128 | waves=4 mt=2 tb=4 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.570 | WIN | 0.758 | 0.805 | 0.275 | WIN | 0.238 | 0.226 | 0.329 | - | - | same |
| gdn_in_tp2_r16 | waves=4 kw=4 mt=4 xsrc=global bar=persist wcp=nt sb=False | 0.424 | WIN | 0.598 | 0.779 | 0.286 | WIN | 0.210 | 0.485 | 0.356 | - | - | same |
| gdn_in_tp2_r32 | waves=4 kw=2 mt=2 depth=4 bar=persist xa=2 bpre=2 sb=False xjunk=True | 0.458 | WIN | 0.615 | 0.795 | 0.282 | WIN | 0.218 | 0.433 | 0.372 | - | 0.922 | WIN |
| gdn_in_tp2_r64 | waves=4 mt=2 tb=2 depth=4 bar=persist map=xcd wcp=nt xa=2 sb=False xjunk=True | 0.499 | WIN | 0.726 | 0.818 | 0.306 | WIN | 0.247 | 0.319 | 0.368 | - | 0.830 | WIN |
| gdn_in_tp2_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.420 | WIN | 0.595 | 0.553 | 0.286 | WIN | 0.217 | 0.520 | 0.357 | - | - | same |
| gdn_out_tp1_r16 | waves=4 kw=4 bar=persist wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.446 | WIN | 0.479 | 0.628 | 0.485 | WIN | 0.336 | 0.473 | 0.388 | - | 0.929 | WIN |
| gdn_out_tp1_r32 | waves=4 kw=4 mt=2 tb=1 depth=4 bar=persist xa=2 bpre=1 sb=False xjunk=True xpad=32 | 0.445 | WIN | 0.506 | 0.660 | 0.485 | WIN | 0.336 | 0.365 | 0.431 | - | - | same |
| gdn_out_tp1_r64 | waves=4 splits=6 depth=4 bar=persist xa=2 sb=False | 0.579 | WIN | 0.875 | 0.929 | 0.632 | WIN | 0.462 | 0.190 | 0.575 | - | - | same |
| gdn_out_tp1_r8 | waves=4 kw=4 depth=3 xsrc=global bar=persist map=xcd sb=False | 0.446 | WIN | 0.444 | 0.615 | 0.473 | WIN | 0.337 | 0.464 | 0.382 | - | - | same |
| gdn_out_tp2_r128 | waves=4 kw=2 mt=2 tb=2 depth=6 bar=persist map=xcd xa=2 sb=False xjunk=True xpad=32 | 0.403 | WIN | 0.353 | 0.563 | 0.437 | WIN | 0.325 | 0.178 | 0.675 | - | 0.796 | WIN |
| gdn_out_tp2_r16 | kw=8 depth=4 bar=persist map=xcd xu=False wcp=nt sb=False xjunk=True xpad=32 | 0.407 | WIN | 0.402 | 0.515 | 0.499 | WIN | 0.309 | 0.349 | 0.794 | - | 0.974 | TIE |
| gdn_out_tp2_r32 | kw=4 tb=1 bar=persist xa=2 xjunk=True | 0.423 | WIN | 0.434 | 0.725 | 0.515 | WIN | 0.341 | 0.268 | 0.846 | - | 0.922 | WIN |
| gdn_out_tp2_r64 | waves=4 kw=2 tb=2 depth=3 bar=persist map=xcd xu=False xa=2 bpre=2 sb=False xjunk=True | 0.408 | WIN | 0.340 | 0.515 | 0.507 | WIN | 0.343 | 0.210 | 0.743 | - | 0.916 | WIN |
| gdn_out_tp2_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.422 | WIN | 0.483 | 0.656 | 0.502 | WIN | 0.319 | 0.343 | 0.837 | - | - | same |
| hc_down_tpx_r128 | waves=4 splits=5 tb=2 depth=3 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.601 | WIN | 0.698 | 0.752 | 0.619 | WIN | 0.450 | 0.054 | 0.723 | - | - | same |
| hc_down_tpx_r16 | waves=4 splits=10 depth=4 bar=persist xu=False wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.613 | WIN | 0.601 | 0.815 | 0.685 | WIN | 0.463 | 0.083 | 0.780 | - | - | same |
| hc_down_tpx_r32 | waves=4 splits=10 tb=1 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.582 | WIN | 0.592 | 0.815 | 0.596 | WIN | 0.431 | 0.084 | 0.774 | - | - | same |
| hc_down_tpx_r64 | waves=4 kw=2 mt=2 splits=5 tb=1 bar=persist xa=2 bpre=1 fg=16 sb=False xpad=32 | 0.538 | WIN | 0.396 | 0.682 | 0.464 | WIN | 0.366 | 0.080 | 0.688 | - | 0.910 | WIN |
| hc_down_tpx_r8 | waves=4 splits=10 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.608 | WIN | 0.913 | 0.930 | 0.690 | WIN | 0.455 | 0.095 | 0.763 | - | - | same |
| hc_up_tpx_r128 | tb=4 depth=4 bar=persist xu=False xa=2 bpre=2 sb=False xjunk=True xpad=32 | 0.515 | WIN | 0.666 | 0.578 | 0.678 | WIN | 0.395 | 0.114 | 0.528 | - | - | same |
| hc_up_tpx_r16 | waves=4 xsrc=global bar=persist xu=False bpre=1 | 0.439 | WIN | 0.498 | 0.670 | 0.667 | WIN | 0.334 | 0.397 | 0.570 | - | - | same |
| hc_up_tpx_r32 | waves=4 depth=6 bar=persist xu=False wcp=nt xa=2 sb=False xjunk=True xpad=32 | 0.457 | WIN | 0.645 | 0.658 | 0.645 | WIN | 0.324 | 0.276 | 0.522 | - | 0.985 | TIE |
| hc_up_tpx_r64 | waves=4 mt=2 tb=2 depth=4 xsrc=global bar=persist xu=False | 0.457 | WIN | 0.605 | 0.638 | 0.689 | WIN | 0.375 | 0.183 | 0.502 | - | - | same |
| hc_up_tpx_r8 | waves=4 xsrc=global bar=persist xu=False wcp=nt | 0.428 | WIN | 0.523 | 0.663 | 0.658 | WIN | 0.339 | 0.382 | 0.575 | - | - | same |
| head_tp1_r16 | waves=4 mt=4 depth=4 bar=persist map=xcd wcp=nt sb=False xjunk=True | 0.730 | WIN | 0.784 | 0.718 | 0.259 | WIN | 0.252 | 0.915 | 0.275 | - | 1.001 | TIE |
| head_tp1_r32 | mt=4 depth=4 bar=persist wcp=nt xa=2 xjunk=True xpad=32 | 0.852 | WIN | 0.940 | 0.900 | 0.246 | WIN | 0.239 | 0.877 | 0.278 | - | - | same |
| head_tp1_r64 | mt=4 depth=4 bar=persist map=xcd wcp=nt xjunk=True xpad=32 | 0.845 | WIN | 0.927 | 0.876 | 0.258 | WIN | 0.253 | 0.788 | 0.328 | - | - | same |
| head_tp1_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt | 0.685 | WIN | 0.729 | 0.687 | 0.237 | WIN | 0.232 | 0.999 | 0.260 | - | - | same |
| head_tp2_r128 | waves=4 mt=4 bar=persist wcp=nt bpre=2 sb=False xpad=32 | 0.786 | WIN | 0.945 | 0.924 | 0.274 | WIN | 0.263 | 0.566 | 0.363 | - | 0.960 | WIN |
| head_tp2_r16 | waves=4 mt=4 depth=4 bar=persist map=xcd wcp=nt xa=2 xjunk=True | 0.655 | WIN | 0.760 | 0.720 | 0.255 | WIN | 0.238 | 0.954 | 0.262 | - | 0.966 | TIE |
| head_tp2_r32 | mt=4 depth=3 bar=persist wcp=nt sb=False xjunk=True | 0.745 | WIN | 0.943 | 0.876 | 0.254 | WIN | 0.238 | 0.877 | 0.274 | - | - | same |
| head_tp2_r64 | mt=4 depth=3 bar=persist wcp=nt xjunk=True xpad=32 | 0.739 | WIN | 0.908 | 0.874 | 0.258 | WIN | 0.246 | 0.802 | 0.309 | - | - | same |
| head_tp2_r8 | mt=4 depth=4 xsrc=global bar=persist wcp=nt bpre=1 sb=False | 0.639 | WIN | 0.719 | 0.734 | 0.245 | WIN | 0.231 | 1.023 | 0.258 | - | - | same |
| qsa_in_tp1_r16 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt sb=False | 0.479 | WIN | 0.616 | 0.820 | 0.283 | WIN | 0.200 | 0.690 | 0.268 | - | - | same |
| qsa_in_tp1_r32 | kw=4 mt=2 depth=4 bar=persist xa=2 sb=False xjunk=True | 0.489 | WIN | 0.645 | 0.846 | 0.288 | WIN | 0.202 | 0.567 | 0.290 | - | - | same |
| qsa_in_tp1_r64 | kw=2 mt=2 tb=2 bar=persist map=xcd wcp=nt sb=False | 0.639 | WIN | 0.843 | 0.863 | 0.336 | WIN | 0.248 | 0.395 | 0.313 | - | 0.991 | TIE |
| qsa_in_tp1_r8 | waves=4 kw=4 mt=4 depth=3 xsrc=global bar=persist wcp=nt | 0.460 | WIN | 0.551 | 0.774 | 0.268 | WIN | 0.187 | 0.783 | 0.251 | - | - | same |
| qsa_in_tp2_r128 | waves=4 mt=2 tb=2 depth=4 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.444 | WIN | 0.523 | 0.613 | 0.331 | WIN | 0.262 | 0.287 | 0.345 | - | 0.824 | WIN |
| qsa_in_tp2_r16 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.363 | WIN | 0.403 | 0.474 | 0.336 | WIN | 0.221 | 0.573 | 0.346 | - | - | same |
| qsa_in_tp2_r32 | waves=4 kw=2 mt=2 tb=1 depth=6 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.394 | WIN | 0.464 | 0.717 | 0.343 | WIN | 0.228 | 0.508 | 0.362 | - | 0.958 | WIN |
| qsa_in_tp2_r64 | waves=4 kw=2 mt=2 tb=2 depth=6 bar=persist map=xcd xa=2 sb=False xjunk=True | 0.401 | WIN | 0.393 | 0.560 | 0.350 | WIN | 0.253 | 0.383 | 0.363 | - | 0.792 | WIN |
| qsa_in_tp2_r8 | waves=4 kw=4 mt=2 xsrc=global bar=persist wcp=nt sb=False | 0.382 | WIN | 0.391 | 0.469 | 0.329 | WIN | 0.237 | 0.634 | 0.332 | - | - | same |
| router_tpx_r128 | waves=4 kw=4 tb=2 depth=4 bar=persist xu=False xa=2 xjunk=True | 0.425 | WIN | 0.480 | 0.580 | 0.323 | WIN | 0.263 | 0.048 | 0.249 | - | - | same |
| router_tpx_r16 | waves=4 kw=4 depth=4 bar=persist xu=False sb=False xjunk=True xpad=32 | 0.375 | WIN | 0.477 | 0.629 | 0.492 | WIN | 0.330 | 0.073 | 0.215 | - | - | same |
| router_tpx_r32 | waves=4 kw=4 tb=1 depth=4 bar=persist xu=False sb=False xjunk=True | 0.358 | WIN | 0.478 | 0.636 | 0.432 | WIN | 0.302 | 0.072 | 0.219 | - | - | same |
| router_tpx_r64 | waves=4 kw=4 tb=1 bar=persist xa=2 sb=False xjunk=True | 0.387 | WIN | 0.382 | 0.670 | 0.374 | WIN | 0.283 | 0.086 | 0.217 | - | 0.941 | WIN |
| router_tpx_r8 | waves=4 kw=4 xsrc=global bar=persist sb=False | 0.389 | WIN | 0.611 | 0.767 | 0.504 | WIN | 0.336 | 0.080 | 0.217 | - | - | same |

Verdict tallies (cells): pick vs base: WIN 51; pick vs fastest replica: WIN 51; pick vs round-6 pick: TIE 6, WIN 13

### Per projection (ranges over rows 5-128 of the per-cell medians)

| projection | pick/base phase | pick/base stream | pick/base launch | pick/fastest replica phase | pick/ft phase | floor/pick (stream) | pick / FT in-situ (same job) | pick / FT in-situ (W4 trace) | pick/round-6 pick (changed cells) |
|---|---|---|---|---|---|---|---|---|---|
| gdn_in_tp1 | 0.54-0.67 | 0.73-0.88 | 0.80-1.06 | 0.25-0.27 | 0.20-0.23 | 0.44-0.62 | 0.23-0.23 | - | 0.90-0.98 |
| gdn_in_tp2 | 0.42-0.57 | 0.59-0.76 | 0.55-0.82 | 0.28-0.31 | 0.21-0.25 | 0.23-0.52 | 0.33-0.37 | - | 0.83-0.92 |
| gdn_out_tp1 | 0.45-0.58 | 0.44-0.88 | 0.61-0.93 | 0.47-0.63 | 0.34-0.46 | 0.19-0.47 | 0.38-0.57 | - | 0.93-0.93 |
| gdn_out_tp2 | 0.40-0.42 | 0.34-0.48 | 0.52-0.72 | 0.44-0.52 | 0.31-0.34 | 0.18-0.35 | 0.68-0.85 | - | 0.80-0.97 |
| hc_down_tpx | 0.54-0.61 | 0.40-0.91 | 0.68-0.93 | 0.46-0.69 | 0.37-0.46 | 0.05-0.10 | 0.69-0.78 | - | 0.91-0.91 |
| hc_up_tpx | 0.43-0.52 | 0.50-0.67 | 0.58-0.67 | 0.65-0.69 | 0.32-0.40 | 0.11-0.40 | 0.50-0.57 | - | 0.98-0.98 |
| head_tp1 | 0.69-0.85 | 0.73-0.94 | 0.69-0.90 | 0.24-0.26 | 0.23-0.25 | 0.79-1.00 | 0.26-0.33 | - | 1.00-1.00 |
| head_tp2 | 0.64-0.79 | 0.72-0.94 | 0.72-0.92 | 0.24-0.27 | 0.23-0.26 | 0.57-1.02 | 0.26-0.36 | - | 0.96-0.97 |
| qsa_in_tp1 | 0.46-0.64 | 0.55-0.84 | 0.77-0.86 | 0.27-0.34 | 0.19-0.25 | 0.40-0.78 | 0.25-0.31 | - | 0.99-0.99 |
| qsa_in_tp2 | 0.36-0.44 | 0.39-0.52 | 0.47-0.72 | 0.33-0.35 | 0.22-0.26 | 0.29-0.63 | 0.33-0.36 | - | 0.79-0.96 |
| router_tpx | 0.36-0.42 | 0.38-0.61 | 0.58-0.77 | 0.32-0.50 | 0.26-0.34 | 0.05-0.09 | 0.21-0.25 | - | 0.94-0.94 |
Same-job FlyTape in-situ: 51 cells, pick phase / FlyTape phase 0.215-0.846; above 1.0: none

Numerics of these runs: per-cell checks 650 (failing 0); suite records 0 (mutants caught 0 / 0); chain checks 0; barrier timeouts 0.

## 3. Library arms (container job C2; launch metric: one call per layer in a graph; LB arm = the fastest fp8 / fp4 finalist of the cell's screen by launch time, which does not depend on the plan table)

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

## 4. Lookback climb (16 rounds, two nodes; a change becomes the pick after a WIN over the previous pick on the second node)

Rounds 1-8 timed pairs and triples of the six best solo levers (`top6`); round 9 every pair of every solo-measured lever
(`all2`); rounds 10-16 also the six best measured pairs extended by every lever (`all2x`).

### 4.1 Climb log (two nodes; a change becomes the pick after a WIN over the previous pick on the second node)

Events: confirmed on n01: 42, confirmed on n07: 32, reverted on n01: 78, reverted on n07: 93, trial on n01: 125, trial on n07: 130

| # | cell | event | node | lever set (trial) | ratio to the previous pick (phase) |
|---|---|---|---|---|---|
| 1 | ('gdn_out', 2, 5) | trial | n01 | map+bpre | 0.975 |
| 2 | ('gdn_out', 2, 64) | trial | n01 | xsrc+bpre+map | 0.884 |
| 3 | ('gdn_out', 2, 128) | trial | n01 | noxu+d3+map | 0.989 |
| 4 | ('hc_down', 0, 5) | trial | n01 | nt+xsrc | 0.977 |
| 5 | ('hc_down', 0, 8) | trial | n01 | s/2+nt | 0.932 |
| 6 | ('hc_down', 0, 16) | trial | n01 | w4+nosb+nt | 0.948 |
| 7 | ('hc_down', 0, 32) | trial | n01 | s/2+mt/2+noxu | 0.851 |
| 8 | ('hc_down', 0, 64) | trial | n01 | s/2+tb/2+w4 | 0.781 |
| 9 | ('hc_down', 0, 128) | trial | n01 | kw*2+bpre | 0.948 |
| 10 | ('hc_up', 0, 5) | trial | n01 | nt | 0.983 |
| 11 | ('hc_up', 0, 8) | trial | n01 | nt+d3 | 0.987 |
| 12 | ('hc_up', 0, 32) | trial | n01 | xsrc+d6 | 0.972 |
| 13 | ('gdn_out', 2, 8) | trial | n07 | nt+bpre | 0.976 |
| 14 | ('gdn_out', 2, 64) | confirmed | n07 |  | 0.897 |
| 15 | ('gdn_out', 2, 64) | trial | n07 | d3+nosb+nt | 0.931 |
| 16 | ('gdn_out', 2, 128) | reverted | n07 |  | 0.998 |
| 17 | ('gdn_out', 1, 5) | trial | n07 | w4+d3+nosb | 0.906 |
| 18 | ('gdn_out', 1, 8) | trial | n07 | w4+d3+nosb | 0.924 |
| 19 | ('gdn_out', 1, 16) | trial | n07 | mt/2+xpad+nt | 0.859 |
| 20 | ('gdn_out', 1, 32) | trial | n07 | tb/2+w4 | 0.758 |
| 21 | ('gdn_out', 1, 64) | trial | n07 | nosb | 0.962 |
| 22 | ('gdn_out', 1, 128) | trial | n07 | noxu+map+bpre | 0.979 |
| 23 | ('hc_down', 0, 5) | reverted | n07 |  | 0.990 |
| 24 | ('hc_down', 0, 8) | confirmed | n07 |  | 0.936 |
| 25 | ('hc_down', 0, 8) | trial | n07 | xa2+fg16+xpad | 0.970 |
| 26 | ('hc_down', 0, 16) | confirmed | n07 |  | 0.952 |
| 27 | ('hc_down', 0, 16) | trial | n07 | mt/2+xjunk | 0.865 |
| 28 | ('hc_down', 0, 32) | confirmed | n07 |  | 0.876 |
| 29 | ('hc_down', 0, 32) | trial | n07 | noxu+xa2 | 0.958 |
| 30 | ('hc_down', 0, 64) | confirmed | n07 |  | 0.772 |
| 31 | ('hc_down', 0, 64) | trial | n07 | mt/2+d2 | 0.917 |
| 32 | ('hc_down', 0, 128) | confirmed | n07 |  | 0.945 |
| 33 | ('hc_down', 0, 128) | trial | n07 | s/2+xa2+d3 | 0.958 |
| 34 | ('hc_up', 0, 32) | confirmed | n07 |  | 0.981 |
| 35 | ('hc_up', 0, 32) | trial | n07 | map+xpad+xjunk | 0.976 |
| 36 | ('hc_up', 0, 64) | trial | n07 | dall | 0.985 |
| 37 | ('router', 0, 128) | trial | n07 | nosb+d3+xpad | 0.979 |
| 38 | ('gdn_in', 2, 5) | trial | n07 | d2+bpre | 0.948 |
| 39 | ('gdn_in', 2, 16) | trial | n07 | bpre+nt | 0.969 |
| 40 | ('gdn_in', 2, 128) | trial | n07 | w4+nosb+map | 0.889 |
| 41 | ('gdn_in', 1, 5) | trial | n07 | map+xpad | 0.976 |
| 42 | ('gdn_in', 1, 16) | trial | n07 | xa2+map | 0.960 |
| 43 | ('gdn_in', 1, 64) | trial | n07 | w4+xpad+map | 0.923 |
| 44 | ('gdn_in', 1, 128) | trial | n07 | map+xpad+xjunk | 0.967 |
| 45 | ('qsa_in', 2, 32) | trial | n07 | xa2+nt+map | 0.968 |
| 46 | ('qsa_in', 2, 128) | trial | n07 | map+nosb | 0.958 |
| 47 | ('qsa_in', 1, 5) | trial | n07 | d2+bpre | 0.968 |
| 48 | ('qsa_in', 1, 8) | trial | n07 | nosb+nt+map | 0.948 |
| 49 | ('qsa_in', 1, 64) | trial | n07 | d3+map+xjunk | 0.970 |
| 50 | ('qsa_in', 1, 128) | trial | n07 | w4+map+noxu | 0.916 |
| 51 | ('head', 2, 32) | trial | n07 | w4 | 0.947 |
| 52 | ('gdn_out', 2, 8) | reverted | n01 |  | 0.998 |
| 53 | ('gdn_out', 2, 32) | trial | n01 | map | 0.972 |
| 54 | ('gdn_out', 2, 64) | confirmed | n01 |  | 0.946 |
| 55 | ('gdn_out', 2, 64) | trial | n01 | nt+noxu+w4 | 0.960 |
| 56 | ('gdn_out', 2, 128) | trial | n01 | noxu+dall+map | 0.933 |
| 57 | ('gdn_out', 1, 5) | confirmed | n01 |  | 0.917 |
| 58 | ('gdn_out', 1, 8) | confirmed | n01 |  | 0.904 |
| 59 | ('gdn_out', 1, 16) | confirmed | n01 |  | 0.887 |
| 60 | ('gdn_out', 1, 32) | confirmed | n01 |  | 0.763 |
| 61 | ('gdn_out', 1, 32) | trial | n01 | xa2+bpre+xpad | 0.927 |
| 62 | ('gdn_out', 1, 64) | confirmed | n01 |  | 0.968 |
| 63 | ('gdn_out', 1, 128) | confirmed | n01 |  | 0.983 |
| 64 | ('hc_down', 0, 5) | trial | n01 | fg16 | 0.983 |
| 65 | ('hc_down', 0, 8) | reverted | n01 |  | 0.975 |
| 66 | ('hc_down', 0, 16) | confirmed | n01 |  | 0.888 |
| 67 | ('hc_down', 0, 16) | trial | n01 | s/2+xa2 | 0.890 |
| 68 | ('hc_down', 0, 32) | confirmed | n01 |  | 0.965 |
| 69 | ('hc_down', 0, 32) | trial | n01 | bpre | 0.984 |
| 70 | ('hc_down', 0, 64) | confirmed | n01 |  | 0.921 |
| 71 | ('hc_down', 0, 64) | trial | n01 | fg16 | 0.992 |
| 72 | ('hc_down', 0, 128) | confirmed | n01 |  | 0.960 |
| 73 | ('hc_down', 0, 128) | trial | n01 | w4+noxu | 0.974 |
| 74 | ('hc_up', 0, 32) | reverted | n01 |  | 0.986 |
| 75 | ('hc_up', 0, 64) | reverted | n01 |  | 0.998 |
| 76 | ('router', 0, 8) | trial | n01 | nt | 0.961 |
| 77 | ('router', 0, 16) | trial | n01 | nt | 0.962 |
| 78 | ('router', 0, 128) | reverted | n01 |  | 0.994 |
| 79 | ('gdn_in', 2, 5) | reverted | n01 |  | 0.988 |
| 80 | ('gdn_in', 2, 8) | trial | n01 | map+d2 | 0.988 |
| 81 | ('gdn_in', 2, 16) | reverted | n01 |  | 1.008 |
| 82 | ('gdn_in', 2, 128) | confirmed | n01 |  | 0.892 |
| 83 | ('gdn_in', 1, 5) | reverted | n01 |  | 0.999 |
| 84 | ('gdn_in', 1, 16) | reverted | n01 |  | 0.977 |
| 85 | ('gdn_in', 1, 64) | confirmed | n01 |  | 0.932 |
| 86 | ('gdn_in', 1, 128) | confirmed | n01 |  | 0.969 |
| 87 | ('qsa_in', 2, 32) | confirmed | n01 |  | 0.972 |
| 88 | ('qsa_in', 2, 128) | confirmed | n01 |  | 0.961 |
| 89 | ('qsa_in', 1, 5) | reverted | n01 |  | 0.997 |
| 90 | ('qsa_in', 1, 8) | reverted | n01 |  | 0.987 |
| 91 | ('qsa_in', 1, 64) | confirmed | n01 |  | 0.983 |
| 92 | ('qsa_in', 1, 128) | confirmed | n01 |  | 0.919 |
| 93 | ('head', 2, 5) | trial | n01 | mt/2 | 0.969 |
| 94 | ('head', 2, 8) | trial | n01 | w4+d3 | 0.965 |
| 95 | ('head', 2, 32) | reverted | n01 |  | 0.990 |
| 96 | ('head', 2, 64) | trial | n01 | noxu+nosb+xjunk | 0.979 |
| 97 | ('head', 2, 128) | trial | n01 | w4+bpre+nosb | 0.939 |
| 98 | ('head', 1, 32) | trial | n01 | nosb+w4 | 0.984 |
| 99 | ('head', 1, 128) | trial | n01 | nt+w4+nosb | 0.945 |
| 100 | ('gdn_out', 2, 5) | reverted | n07 |  | 0.979 |
| 101 | ('gdn_out', 2, 16) | trial | n07 | noxu+nt | 0.973 |
| 102 | ('gdn_out', 2, 32) | reverted | n07 |  | 0.991 |
| 103 | ('gdn_out', 2, 64) | confirmed | n07 |  | 0.964 |
| 104 | ('gdn_out', 2, 128) | confirmed | n07 |  | 0.942 |
| 105 | ('gdn_out', 1, 32) | confirmed | n07 |  | 0.950 |
| 106 | ('gdn_out', 1, 32) | trial | n07 | map+d2 | 0.979 |
| 107 | ('gdn_out', 1, 128) | trial | n07 | d6+nt | 0.980 |
| 108 | ('hc_down', 0, 5) | reverted | n07 |  | 1.000 |
| 109 | ('hc_down', 0, 8) | trial | n07 | xa2+nt | 0.986 |
| 110 | ('hc_down', 0, 16) | confirmed | n07 |  | 0.908 |
| 111 | ('hc_down', 0, 16) | trial | n07 | dall+fg16 | 0.990 |
| 112 | ('hc_down', 0, 32) | reverted | n07 |  | 1.000 |
| 113 | ('hc_down', 0, 64) | reverted | n07 |  |  |
| 114 | ('hc_down', 0, 128) | confirmed | n07 |  | 0.982 |
| 115 | ('hc_down', 0, 128) | trial | n07 | kw/2+xpad | 0.920 |
| 116 | ('hc_up', 0, 5) | reverted | n07 |  | 0.996 |
| 117 | ('hc_up', 0, 8) | reverted | n07 |  | 1.005 |
| 118 | ('hc_up', 0, 16) | trial | n07 | nt | 0.989 |
| 119 | ('hc_up', 0, 128) | trial | n07 | xa2+bpre+nosb | 0.979 |
| 120 | ('router', 0, 8) | reverted | n07 |  | 0.998 |
| 121 | ('router', 0, 16) | reverted | n07 |  | 0.998 |
| 122 | ('router', 0, 32) | trial | n07 | bpre | 0.988 |
| 123 | ('gdn_in', 2, 8) | reverted | n07 |  | 1.001 |
| 124 | ('gdn_in', 1, 16) | trial | n07 | xa2+map+xpad | 0.967 |
| 125 | ('qsa_in', 2, 5) | trial | n07 | bpre+map | 0.968 |
| 126 | ('qsa_in', 2, 8) | trial | n07 | map+bpre | 0.971 |
| 127 | ('qsa_in', 2, 16) | trial | n07 | bpre | 0.981 |
| 128 | ('qsa_in', 2, 64) | trial | n07 | xjunk+map+xpad | 0.982 |
| 129 | ('head', 2, 5) | reverted | n07 |  | 0.981 |
| 130 | ('head', 2, 8) | reverted | n07 |  | 0.979 |
| 131 | ('head', 2, 64) | reverted | n07 |  | 0.977 |
| 132 | ('head', 2, 128) | reverted | n07 |  | 0.946 |
| 133 | ('head', 1, 32) | reverted | n07 |  | 1.006 |
| 134 | ('head', 1, 128) | reverted | n07 |  | 0.948 |
| 135 | ('gdn_out', 2, 5) | trial | n01 | map+bpre | 0.954 |
| 136 | ('gdn_out', 2, 8) | trial | n01 | map+nt+bpre | 0.974 |
| 137 | ('gdn_out', 2, 16) | reverted | n01 |  | 0.967 |
| 138 | ('gdn_out', 1, 32) | reverted | n01 |  | 0.990 |
| 139 | ('gdn_out', 1, 128) | reverted | n01 |  | 0.982 |
| 140 | ('hc_down', 0, 5) | trial | n01 | nt+fg32+xa2 | 0.979 |
| 141 | ('hc_down', 0, 8) | confirmed | n01 |  | 0.980 |
| 142 | ('hc_down', 0, 8) | trial | n01 | noxu+nt+xpad | 0.989 |
| 143 | ('hc_down', 0, 16) | reverted | n01 |  | 0.989 |
| 144 | ('hc_down', 0, 32) | trial | n01 | d2+noxu | 0.965 |
| 145 | ('hc_down', 0, 128) | confirmed | n01 |  | 0.921 |
| 146 | ('hc_up', 0, 5) | trial | n01 | nt | 0.988 |
| 147 | ('hc_up', 0, 16) | reverted | n01 |  | 0.997 |
| 148 | ('hc_up', 0, 128) | confirmed | n01 |  | 0.983 |
| 149 | ('router', 0, 8) | trial | n01 | nt+map | 0.954 |
| 150 | ('router', 0, 16) | trial | n01 | nt | 0.965 |
| 151 | ('router', 0, 32) | reverted | n01 |  | 1.000 |
| 152 | ('router', 0, 128) | trial | n01 | nosb+d3+xpad | 0.979 |
| 153 | ('gdn_in', 2, 5) | trial | n01 | map+d3+bpre | 0.953 |
| 154 | ('gdn_in', 2, 16) | trial | n01 | map+d3+xsrc | 0.968 |
| 155 | ('gdn_in', 1, 5) | trial | n01 | map+xpad | 0.982 |
| 156 | ('gdn_in', 1, 16) | reverted | n01 |  | 0.975 |
| 157 | ('qsa_in', 2, 5) | reverted | n01 |  | 0.990 |
| 158 | ('qsa_in', 2, 8) | reverted | n01 |  | 0.988 |
| 159 | ('qsa_in', 2, 16) | reverted | n01 |  | 1.003 |
| 160 | ('qsa_in', 2, 64) | reverted | n01 |  | 0.988 |
| 161 | ('qsa_in', 1, 8) | trial | n01 | nosb+map | 0.970 |
| 162 | ('head', 2, 8) | trial | n01 | w4 | 0.971 |
| 163 | ('head', 1, 32) | trial | n01 | d3+bpre+map | 0.984 |
| 164 | ('gdn_out', 2, 5) | confirmed | n07 |  | 0.983 |
| 165 | ('gdn_out', 2, 8) | reverted | n07 |  | 0.989 |
| 166 | ('gdn_out', 2, 16) | trial | n07 | noxu+nt+xpad | 0.969 |
| 167 | ('gdn_out', 2, 64) | trial | n07 | map+xjunk | 0.980 |
| 168 | ('gdn_out', 1, 16) | trial | n07 | map+bpre | 0.982 |
| 169 | ('gdn_out', 1, 32) | trial | n07 | xjunk | 0.981 |
| 170 | ('hc_down', 0, 5) | reverted | n07 |  | 0.991 |
| 171 | ('hc_down', 0, 8) | reverted | n07 |  | 0.995 |
| 172 | ('hc_down', 0, 16) | trial | n07 | nt+dall | 0.989 |
| 173 | ('hc_down', 0, 32) | reverted | n07 |  | 1.009 |
| 174 | ('hc_up', 0, 5) | reverted | n07 |  | 0.997 |
| 175 | ('hc_up', 0, 16) | trial | n07 | nt | 0.986 |
| 176 | ('router', 0, 8) | reverted | n07 |  | 1.001 |
| 177 | ('router', 0, 16) | reverted | n07 |  | 0.999 |
| 178 | ('router', 0, 128) | reverted | n07 |  | 0.999 |
| 179 | ('gdn_in', 2, 5) | reverted | n07 |  | 0.990 |
| 180 | ('gdn_in', 2, 8) | trial | n07 | bpre | 0.983 |
| 181 | ('gdn_in', 2, 16) | reverted | n07 |  | 0.992 |
| 182 | ('gdn_in', 1, 5) | reverted | n07 |  | 0.995 |
| 183 | ('gdn_in', 1, 16) | trial | n07 | xa2+map+xpad | 0.962 |
| 184 | ('qsa_in', 2, 5) | trial | n07 | map | 0.964 |
| 185 | ('qsa_in', 2, 8) | trial | n07 | map | 0.958 |
| 186 | ('qsa_in', 2, 16) | trial | n07 | map | 0.965 |
| 187 | ('qsa_in', 1, 8) | reverted | n07 |  | 0.967 |
| 188 | ('head', 2, 8) | reverted | n07 |  | 0.983 |
| 189 | ('head', 1, 32) | reverted | n07 |  | 0.983 |
| 190 | ('head', 1, 128) | trial | n07 | nosb+w4+map | 0.942 |
| 191 | ('gdn_out', 2, 8) | trial | n01 | map | 0.982 |
| 192 | ('gdn_out', 2, 16) | confirmed | n01 |  | 0.976 |
| 193 | ('gdn_out', 2, 32) | trial | n01 | nt+xpad+dall | 0.985 |
| 194 | ('gdn_out', 2, 64) | reverted | n01 |  | 0.998 |
| 195 | ('gdn_out', 1, 16) | reverted | n01 |  | 0.996 |
| 196 | ('gdn_out', 1, 32) | reverted | n01 |  | 0.997 |
| 197 | ('hc_down', 0, 5) | trial | n01 | nt+fg32+xa2 | 0.974 |
| 198 | ('hc_down', 0, 16) | reverted | n01 |  | 0.993 |
| 199 | ('hc_down', 0, 32) | trial | n01 | noxu+d2 | 0.969 |
| 200 | ('hc_up', 0, 8) | trial | n01 | bpre+d6+mt*2 | 0.989 |
| 201 | ('hc_up', 0, 16) | reverted | n01 |  | 0.998 |
| 202 | ('router', 0, 8) | trial | n01 | nt+bpre | 0.980 |
| 203 | ('router', 0, 128) | trial | n01 | nosb+d3+xpad | 0.978 |
| 204 | ('gdn_in', 2, 5) | trial | n01 | d2+map | 0.950 |
| 205 | ('gdn_in', 2, 8) | reverted | n01 |  | 1.000 |
| 206 | ('gdn_in', 2, 16) | trial | n01 | d3+xsrc | 0.969 |
| 207 | ('gdn_in', 2, 128) | trial | n01 | xpad+xjunk+map | 0.985 |
| 208 | ('gdn_in', 1, 5) | trial | n01 | bpre+xpad | 0.982 |
| 209 | ('gdn_in', 1, 16) | confirmed | n01 |  | 0.969 |
| 210 | ('qsa_in', 2, 5) | reverted | n01 |  | 0.989 |
| 211 | ('qsa_in', 2, 8) | reverted | n01 |  | 0.991 |
| 212 | ('qsa_in', 2, 16) | reverted | n01 |  | 0.985 |
| 213 | ('head', 2, 8) | trial | n01 | d3+w4 | 0.972 |
| 214 | ('head', 2, 32) | trial | n01 | w4+xpad | 0.938 |
| 215 | ('head', 1, 16) | trial | n01 | d3+xa2+nosb | 0.977 |
| 216 | ('head', 1, 128) | confirmed | n01 |  | 0.947 |
| 217 | ('gdn_out', 2, 8) | reverted | n07 |  | 0.991 |
| 218 | ('gdn_out', 2, 16) | trial | n07 | bpre | 0.981 |
| 219 | ('gdn_out', 2, 32) | reverted | n07 |  | 0.985 |
| 220 | ('gdn_out', 2, 64) | trial | n07 | map+xjunk+d2 | 0.981 |
| 221 | ('gdn_out', 1, 16) | trial | n07 | nosb+d2+w4 | 0.924 |
| 222 | ('gdn_out', 1, 32) | trial | n07 | xjunk+map+d2 | 0.978 |
| 223 | ('hc_down', 0, 5) | reverted | n07 |  | 0.993 |
| 224 | ('hc_down', 0, 8) | trial | n07 | fg16 | 0.984 |
| 225 | ('hc_down', 0, 16) | trial | n07 | dall | 0.971 |
| 226 | ('hc_down', 0, 32) | reverted | n07 |  | 1.003 |
| 227 | ('hc_down', 0, 64) | trial | n07 | fg16 | 0.993 |
| 228 | ('hc_up', 0, 5) | trial | n07 | nt | 0.983 |
| 229 | ('hc_up', 0, 8) | reverted | n07 |  | 1.011 |
| 230 | ('hc_up', 0, 16) | trial | n07 | nt+map | 0.981 |
| 231 | ('hc_up', 0, 128) | trial | n07 | xpad | 0.989 |
| 232 | ('router', 0, 8) | reverted | n07 |  | 0.998 |
| 233 | ('router', 0, 32) | trial | n07 | bpre | 0.986 |
| 234 | ('router', 0, 64) | trial | n07 | bpre | 0.981 |
| 235 | ('router', 0, 128) | reverted | n07 |  | 0.995 |
| 236 | ('gdn_in', 2, 5) | reverted | n07 |  | 0.976 |
| 237 | ('gdn_in', 2, 16) | reverted | n07 |  | 0.992 |
| 238 | ('gdn_in', 2, 64) | trial | n07 | xpad | 0.979 |
| 239 | ('gdn_in', 2, 128) | reverted | n07 |  | 1.010 |
| 240 | ('gdn_in', 1, 5) | reverted | n07 |  | 0.998 |
| 241 | ('qsa_in', 2, 128) | trial | n07 | xpad | 0.985 |
| 242 | ('qsa_in', 1, 8) | trial | n07 | d2+map | 0.968 |
| 243 | ('qsa_in', 1, 32) | trial | n07 | nt+map+xjunk | 0.968 |
| 244 | ('head', 2, 8) | reverted | n07 |  | 0.978 |
| 245 | ('head', 2, 32) | reverted | n07 |  | 0.987 |
| 246 | ('head', 1, 16) | reverted | n07 |  | 0.983 |
| 247 | ('gdn_out', 2, 5) | trial | n01 | nt | 0.980 |
| 248 | ('gdn_out', 2, 16) | reverted | n01 |  | 1.001 |
| 249 | ('gdn_out', 2, 64) | reverted | n01 |  | 1.000 |
| 250 | ('gdn_out', 2, 128) | trial | n01 | kw/2+tb/2 | 0.909 |
| 251 | ('gdn_out', 1, 16) | confirmed | n01 |  | 0.935 |
| 252 | ('gdn_out', 1, 16) | trial | n01 | map+xjunk | 0.982 |
| 253 | ('gdn_out', 1, 32) | reverted | n01 |  | 0.992 |
| 254 | ('gdn_out', 1, 128) | trial | n01 | tb/2+mt*2 | 0.915 |
| 255 | ('hc_down', 0, 5) | trial | n01 | xsrc+s/2 | 0.907 |
| 256 | ('hc_down', 0, 8) | reverted | n01 |  | 1.000 |
| 257 | ('hc_down', 0, 16) | reverted | n01 |  | 0.989 |
| 258 | ('hc_down', 0, 32) | trial | n01 | noxu+d3 | 0.965 |
| 259 | ('hc_down', 0, 64) | reverted | n01 |  |  |
| 260 | ('hc_up', 0, 5) | reverted | n01 |  | 0.996 |
| 261 | ('hc_up', 0, 8) | trial | n01 | nosb+xsrc | 0.986 |
| 262 | ('hc_up', 0, 16) | reverted | n01 |  | 1.004 |
| 263 | ('hc_up', 0, 32) | trial | n01 | xpad+map | 0.978 |
| 264 | ('hc_up', 0, 128) | reverted | n01 |  |  |
| 265 | ('router', 0, 5) | trial | n01 | nt+map | 0.973 |
| 266 | ('router', 0, 32) | reverted | n01 |  | 1.000 |
| 267 | ('router', 0, 64) | reverted | n01 |  | 1.002 |
| 268 | ('gdn_in', 2, 32) | trial | n01 | w4+kw/2 | 0.953 |
| 269 | ('gdn_in', 2, 64) | reverted | n01 |  | 0.991 |
| 270 | ('gdn_in', 1, 32) | trial | n01 | w4+kw/2 | 0.957 |
| 271 | ('gdn_in', 1, 128) | trial | n01 | d3+bpre | 0.989 |
| 272 | ('qsa_in', 2, 5) | trial | n01 | bpre+map+nt | 0.969 |
| 273 | ('qsa_in', 2, 8) | trial | n01 | nt+bpre+map | 0.967 |
| 274 | ('qsa_in', 2, 16) | trial | n01 | nt+bpre+map | 0.965 |
| 275 | ('qsa_in', 2, 64) | trial | n01 | tb/2+kw/2 | 0.904 |
| 276 | ('qsa_in', 2, 128) | reverted | n01 |  | 0.992 |
| 277 | ('qsa_in', 1, 8) | reverted | n01 |  | 1.003 |
| 278 | ('qsa_in', 1, 32) | reverted | n01 |  | 0.979 |
| 279 | ('head', 2, 16) | trial | n01 | xa2+w4 | 0.956 |
| 280 | ('gdn_out', 2, 5) | reverted | n07 |  | 0.999 |
| 281 | ('gdn_out', 2, 8) | trial | n07 | nt | 0.979 |
| 282 | ('gdn_out', 2, 32) | trial | n07 | kw/2+tb/2+xa2 | 0.943 |
| 283 | ('gdn_out', 2, 128) | confirmed | n07 |  | 0.891 |
| 284 | ('gdn_out', 2, 128) | trial | n07 | xa2+noxu+d2 | 0.913 |
| 285 | ('gdn_out', 1, 16) | reverted | n07 |  | 0.987 |
| 286 | ('gdn_out', 1, 32) | trial | n07 | map+nt+d2 | 0.970 |
| 287 | ('gdn_out', 1, 64) | trial | n07 | bpre | 0.991 |
| 288 | ('gdn_out', 1, 128) | confirmed | n07 |  | 0.902 |
| 289 | ('gdn_out', 1, 128) | trial | n07 | xa2+noxu | 0.955 |
| 290 | ('hc_down', 0, 5) | confirmed | n07 |  | 0.922 |
| 291 | ('hc_down', 0, 5) | trial | n07 | mt*2+kw*2+nt | 0.948 |
| 292 | ('hc_down', 0, 8) | trial | n07 | nt+noxu | 0.975 |
| 293 | ('hc_down', 0, 16) | trial | n07 | dall | 0.966 |
| 294 | ('hc_down', 0, 32) | reverted | n07 |  | 0.984 |
| 295 | ('hc_down', 0, 64) | trial | n07 | mt*2+kw*2+nosb | 0.956 |
| 296 | ('hc_up', 0, 5) | trial | n07 | nt+mt*2+d2 | 0.981 |
| 297 | ('hc_up', 0, 8) | reverted | n07 |  | 0.998 |
| 298 | ('hc_up', 0, 32) | reverted | n07 |  | 0.987 |
| 299 | ('hc_up', 0, 64) | trial | n07 | xsrc+nosb+dall | 0.966 |
| 300 | ('router', 0, 5) | reverted | n07 |  | 1.001 |
| 301 | ('router', 0, 8) | trial | n07 | nt+map | 0.971 |
| 302 | ('router', 0, 32) | trial | n07 | bpre | 0.984 |
| 303 | ('router', 0, 64) | trial | n07 | xsrc+noxu | 0.962 |
| 304 | ('router', 0, 128) | trial | n07 | map+nosb+d3 | 0.988 |
| 305 | ('gdn_in', 2, 32) | confirmed | n07 |  | 0.952 |
| 306 | ('gdn_in', 2, 64) | trial | n07 | kw/2+tb/2+nt | 0.871 |
| 307 | ('gdn_in', 1, 5) | trial | n07 | w4+kw/2+bpre | 0.960 |
| 308 | ('gdn_in', 1, 16) | trial | n07 | bpre | 0.981 |
| 309 | ('gdn_in', 1, 32) | confirmed | n07 |  | 0.946 |
| 310 | ('gdn_in', 1, 32) | trial | n07 | nt+nosb+bpre | 0.947 |
| 311 | ('gdn_in', 1, 128) | reverted | n07 |  | 0.992 |
| 312 | ('qsa_in', 2, 5) | reverted | n07 |  | 1.001 |
| 313 | ('qsa_in', 2, 8) | reverted | n07 |  | 1.007 |
| 314 | ('qsa_in', 2, 16) | reverted | n07 |  | 1.018 |
| 315 | ('qsa_in', 2, 32) | trial | n07 | kw/2+tb/2+d6 | 0.959 |
| 316 | ('qsa_in', 2, 64) | confirmed | n07 |  | 0.922 |
| 317 | ('qsa_in', 2, 64) | trial | n07 | xa2+map+d6 | 0.912 |
| 318 | ('qsa_in', 2, 128) | trial | n07 | kw/2+tb/2+w4 | 0.837 |
| 319 | ('qsa_in', 1, 64) | trial | n07 | tb/2+kw/2+d2 | 0.954 |
| 320 | ('head', 2, 16) | reverted | n07 |  | 0.982 |
| 321 | ('gdn_out', 2, 8) | reverted | n01 |  | 1.000 |
| 322 | ('gdn_out', 2, 16) | trial | n01 | xa2 | 0.972 |
| 323 | ('gdn_out', 2, 32) | confirmed | n01 |  | 0.962 |
| 324 | ('gdn_out', 2, 64) | trial | n01 | kw/2+mt/2+xa2 | 0.969 |
| 325 | ('gdn_out', 2, 128) | confirmed | n01 |  | 0.918 |
| 326 | ('gdn_out', 1, 32) | reverted | n01 |  | 0.983 |
| 327 | ('gdn_out', 1, 64) | reverted | n01 |  | 0.999 |
| 328 | ('gdn_out', 1, 128) | confirmed | n01 |  | 0.953 |
| 329 | ('hc_down', 0, 5) | confirmed | n01 |  | 0.968 |
| 330 | ('hc_down', 0, 5) | trial | n01 | nt+d2+noxu | 0.968 |
| 331 | ('hc_down', 0, 8) | reverted | n01 |  | 0.997 |
| 332 | ('hc_down', 0, 16) | reverted | n01 |  | 0.990 |
| 333 | ('hc_down', 0, 32) | trial | n01 | d3+noxu+fg16 | 0.992 |
| 334 | ('hc_down', 0, 64) | confirmed | n01 |  | 0.965 |
| 335 | ('hc_down', 0, 64) | trial | n01 | s/2+bpre | 0.961 |
| 336 | ('hc_up', 0, 5) | reverted | n01 |  | 1.010 |
| 337 | ('hc_up', 0, 32) | trial | n01 | nosb+xa2+xpad | 0.960 |
| 338 | ('hc_up', 0, 64) | reverted | n01 |  | 0.994 |
| 339 | ('router', 0, 5) | trial | n01 | nt+bpre | 0.953 |
| 340 | ('router', 0, 8) | reverted | n01 |  | 1.004 |
| 341 | ('router', 0, 16) | trial | n01 | nt | 0.965 |
| 342 | ('router', 0, 32) | reverted | n01 |  | 0.999 |
| 343 | ('router', 0, 64) | confirmed | n01 |  | 0.972 |
| 344 | ('router', 0, 64) | trial | n01 | xa2+noxu | 0.961 |
| 345 | ('router', 0, 128) | reverted | n01 |  | 1.001 |
| 346 | ('gdn_in', 2, 5) | trial | n01 | d2+bpre | 0.966 |
| 347 | ('gdn_in', 2, 64) | confirmed | n01 |  | 0.881 |
| 348 | ('gdn_in', 2, 64) | trial | n01 | w4+map+nt | 0.930 |
| 349 | ('gdn_in', 1, 5) | confirmed | n01 |  | 0.963 |
| 350 | ('gdn_in', 1, 5) | trial | n01 | mt/2+kw/2 | 0.944 |
| 351 | ('gdn_in', 1, 8) | trial | n01 | w4+kw/2+d2 | 0.938 |
| 352 | ('gdn_in', 1, 16) | reverted | n01 |  | 1.001 |
| 353 | ('gdn_in', 1, 32) | confirmed | n01 |  | 0.936 |
| 354 | ('gdn_in', 1, 128) | trial | n01 | xa2+w4+nt | 0.966 |
| 355 | ('qsa_in', 2, 32) | confirmed | n01 |  | 0.970 |
| 356 | ('qsa_in', 2, 64) | confirmed | n01 |  | 0.909 |
| 357 | ('qsa_in', 2, 128) | confirmed | n01 |  | 0.829 |
| 358 | ('qsa_in', 1, 8) | trial | n01 | nosb+nt | 0.953 |
| 359 | ('qsa_in', 1, 64) | confirmed | n01 |  | 0.963 |
| 360 | ('head', 2, 8) | trial | n01 | w4 | 0.974 |
| 361 | ('head', 2, 16) | trial | n01 | xa2+w4+map | 0.958 |
| 362 | ('head', 2, 128) | trial | n01 | noxu+nosb+w4 | 0.953 |
| 363 | ('head', 1, 16) | trial | n01 | nosb+w4+noxu | 0.971 |
| 364 | ('head', 1, 32) | trial | n01 | w4+nosb | 0.978 |
| 365 | ('gdn_out', 2, 5) | trial | n07 | nt | 0.979 |
| 366 | ('gdn_out', 2, 8) | trial | n07 | map | 0.965 |
| 367 | ('gdn_out', 2, 16) | reverted | n07 |  | 1.007 |
| 368 | ('gdn_out', 2, 64) | confirmed | n07 |  | 0.950 |
| 369 | ('gdn_out', 1, 16) | trial | n07 | map+xjunk | 0.981 |
| 370 | ('gdn_out', 1, 32) | trial | n07 | map+nt+d6 | 0.969 |
| 371 | ('hc_down', 0, 5) | confirmed | n07 |  | 0.979 |
| 372 | ('hc_down', 0, 5) | trial | n07 | nosb+nt+bpre | 0.972 |
| 373 | ('hc_down', 0, 16) | trial | n07 | dall+bpre | 0.979 |
| 374 | ('hc_down', 0, 32) | reverted | n07 |  | 0.995 |
| 375 | ('hc_down', 0, 64) | confirmed | n07 |  | 0.963 |
| 376 | ('hc_up', 0, 5) | trial | n07 | nt | 0.991 |
| 377 | ('hc_up', 0, 32) | confirmed | n07 |  | 0.972 |
| 378 | ('hc_up', 0, 64) | trial | n07 | xsrc+nosb+map | 0.980 |
| 379 | ('router', 0, 5) | reverted | n07 |  | 0.998 |
| 380 | ('router', 0, 16) | reverted | n07 |  | 0.998 |
| 381 | ('router', 0, 32) | trial | n07 | bpre | 0.979 |
| 382 | ('router', 0, 64) | confirmed | n07 |  | 0.968 |
| 383 | ('gdn_in', 2, 5) | reverted | n07 |  | 0.981 |
| 384 | ('gdn_in', 2, 64) | confirmed | n07 |  | 0.932 |
| 385 | ('gdn_in', 1, 5) | confirmed | n07 |  | 0.963 |
| 386 | ('gdn_in', 1, 8) | reverted | n07 |  | 0.966 |
| 387 | ('gdn_in', 1, 16) | trial | n07 | w4+kw/2+d2 | 0.964 |
| 388 | ('gdn_in', 1, 128) | confirmed | n07 |  | 0.970 |
| 389 | ('gdn_in', 1, 128) | trial | n07 | bpre | 0.986 |
| 390 | ('qsa_in', 2, 8) | trial | n07 | bpre | 0.975 |
| 391 | ('qsa_in', 2, 16) | trial | n07 | bpre | 0.979 |
| 392 | ('qsa_in', 1, 5) | trial | n07 | d3 | 0.985 |
| 393 | ('qsa_in', 1, 8) | reverted | n07 |  | 0.983 |
| 394 | ('head', 2, 8) | reverted | n07 |  | 0.977 |
| 395 | ('head', 2, 16) | confirmed | n07 |  | 0.955 |
| 396 | ('head', 2, 128) | reverted | n07 |  | 0.964 |
| 397 | ('head', 1, 16) | confirmed | n07 |  | 0.974 |
| 398 | ('head', 1, 32) | reverted | n07 |  | 0.987 |
| 399 | ('gdn_out', 2, 5) | reverted | n01 |  | 0.998 |
| 400 | ('gdn_out', 2, 8) | reverted | n01 |  | 0.994 |
| 401 | ('gdn_out', 2, 32) | trial | n01 | noxu+nosb+d2 | 0.966 |
| 402 | ('gdn_out', 1, 16) | reverted | n01 |  | 0.990 |
| 403 | ('gdn_out', 1, 32) | reverted | n01 |  | 0.980 |
| 404 | ('hc_down', 0, 5) | confirmed | n01 |  | 0.981 |
| 405 | ('hc_down', 0, 5) | trial | n01 | fg16 | 0.987 |
| 406 | ('hc_down', 0, 8) | trial | n01 | nt+xpad+noxu | 0.978 |
| 407 | ('hc_down', 0, 16) | reverted | n01 |  | 0.990 |
| 408 | ('hc_down', 0, 64) | trial | n01 | xjunk | 0.994 |
| 409 | ('hc_up', 0, 5) | reverted | n01 |  | 0.998 |
| 410 | ('hc_up', 0, 64) | reverted | n01 |  | 1.000 |
| 411 | ('router', 0, 5) | trial | n01 | nt+map | 0.971 |
| 412 | ('router', 0, 8) | trial | n01 | nt+bpre | 0.957 |
| 413 | ('router', 0, 16) | trial | n01 | nt | 0.960 |
| 414 | ('router', 0, 32) | reverted | n01 |  | 1.001 |
| 415 | ('router', 0, 128) | trial | n01 | map+nosb+xpad | 0.982 |
| 416 | ('gdn_in', 1, 8) | trial | n01 | w4+kw/2 | 0.951 |
| 417 | ('gdn_in', 1, 16) | reverted | n01 |  | 0.974 |
| 418 | ('gdn_in', 1, 128) | confirmed | n01 |  | 0.988 |
| 419 | ('qsa_in', 2, 5) | trial | n01 | map | 0.957 |
| 420 | ('qsa_in', 2, 8) | reverted | n01 |  | 1.000 |
| 421 | ('qsa_in', 2, 16) | reverted | n01 |  | 1.000 |
| 422 | ('qsa_in', 1, 5) | reverted | n01 |  | 0.989 |
| 423 | ('qsa_in', 1, 8) | trial | n01 | nosb+nt+map | 0.953 |
| 424 | ('head', 2, 5) | trial | n01 | w4+mt/2+d3 | 0.970 |
| 425 | ('head', 2, 8) | trial | n01 | w4+xsrc+d3 | 0.961 |
| 426 | ('head', 2, 32) | trial | n01 | w4+xpad | 0.964 |
| 427 | ('head', 2, 128) | trial | n01 | nosb+noxu+w4 | 0.949 |
| 428 | ('head', 1, 8) | trial | n01 | noxu+d2+w4 | 0.976 |
| 429 | ('head', 1, 32) | trial | n01 | w4+nosb | 0.982 |
| 430 | ('gdn_out', 2, 32) | confirmed | n07 |  | 0.959 |
| 431 | ('gdn_out', 2, 32) | trial | n07 | mt*2+w4+nosb | 0.929 |
| 432 | ('gdn_out', 2, 128) | trial | n07 | w4+mt*2+d6 | 0.968 |
| 433 | ('gdn_out', 1, 16) | trial | n07 | map | 0.986 |
| 434 | ('hc_down', 0, 5) | reverted | n07 |  | 1.000 |
| 435 | ('hc_down', 0, 8) | reverted | n07 |  | 0.994 |
| 436 | ('hc_down', 0, 64) | reverted | n07 |  | 0.999 |
| 437 | ('hc_up', 0, 16) | trial | n07 | nosb+xsrc+map | 0.985 |
| 438 | ('router', 0, 5) | reverted | n07 |  | 0.995 |
| 439 | ('router', 0, 8) | reverted | n07 |  | 0.999 |
| 440 | ('router', 0, 16) | reverted | n07 |  | 1.000 |
| 441 | ('router', 0, 32) | trial | n07 | bpre | 0.977 |
| 442 | ('router', 0, 128) | reverted | n07 |  | 0.999 |
| 443 | ('gdn_in', 1, 8) | reverted | n07 |  | 0.981 |
| 444 | ('gdn_in', 1, 16) | trial | n07 | w4+kw/2+d6 | 0.957 |
| 445 | ('qsa_in', 2, 5) | reverted | n07 |  | 0.997 |
| 446 | ('qsa_in', 2, 8) | trial | n07 | map | 0.955 |
| 447 | ('qsa_in', 2, 16) | trial | n07 | bpre | 0.967 |
| 448 | ('qsa_in', 2, 32) | trial | n07 | nt | 0.980 |
| 449 | ('qsa_in', 1, 8) | reverted | n07 |  | 0.978 |
| 450 | ('qsa_in', 1, 32) | trial | n07 | w4+kw/2+nt | 0.960 |
| 451 | ('head', 2, 5) | confirmed | n07 |  | 0.967 |
| 452 | ('head', 2, 8) | reverted | n07 |  | 0.968 |
| 453 | ('head', 2, 32) | reverted | n07 |  | 0.981 |
| 454 | ('head', 2, 128) | reverted | n07 |  | 0.961 |
| 455 | ('head', 1, 8) | reverted | n07 |  | 0.993 |
| 456 | ('head', 1, 32) | reverted | n07 |  | 0.983 |
| 457 | ('gdn_out', 2, 8) | trial | n01 | map | 0.974 |
| 458 | ('gdn_out', 2, 32) | reverted | n01 |  | 0.965 |
| 459 | ('gdn_out', 2, 128) | confirmed | n01 |  | 0.975 |
| 460 | ('gdn_out', 1, 16) | reverted | n01 |  | 0.990 |
| 461 | ('gdn_out', 1, 32) | trial | n01 | map+nt | 0.975 |
| 462 | ('hc_down', 0, 16) | trial | n01 | nt+dall+bpre | 0.981 |
| 463 | ('hc_down', 0, 128) | trial | n01 | s*2 | 0.992 |
| 464 | ('hc_up', 0, 8) | trial | n01 | nosb+xsrc+mt*2 | 0.979 |
| 465 | ('hc_up', 0, 16) | reverted | n01 |  | 0.991 |
| 466 | ('hc_up', 0, 128) | trial | n01 | dall | 0.990 |
| 467 | ('router', 0, 32) | reverted | n01 |  | 1.000 |
| 468 | ('gdn_in', 2, 5) | trial | n01 | d3 | 0.978 |
| 469 | ('gdn_in', 1, 8) | trial | n01 | w4+kw/2+d6 | 0.925 |
| 470 | ('gdn_in', 1, 16) | reverted | n01 |  | 0.976 |
| 471 | ('qsa_in', 2, 5) | trial | n01 | bpre | 0.968 |
| 472 | ('qsa_in', 2, 8) | reverted | n01 |  | 0.998 |
| 473 | ('qsa_in', 2, 16) | reverted | n01 |  | 0.999 |
| 474 | ('qsa_in', 2, 32) | confirmed | n01 |  | 0.984 |
| 475 | ('qsa_in', 1, 8) | trial | n01 | nosb+map+nt | 0.967 |
| 476 | ('qsa_in', 1, 32) | reverted | n01 |  | 0.981 |
| 477 | ('head', 2, 8) | trial | n01 | w4+d3+xsrc | 0.957 |
| 478 | ('head', 2, 128) | trial | n01 | bpre+w4+nosb | 0.947 |
| 479 | ('gdn_out', 2, 8) | reverted | n07 |  | 0.991 |
| 480 | ('gdn_out', 2, 16) | trial | n07 | dall | 0.986 |
| 481 | ('gdn_out', 2, 32) | trial | n07 | mt*2+w4+nosb | 0.961 |
| 482 | ('gdn_out', 1, 32) | reverted | n07 |  | 0.984 |
| 483 | ('hc_down', 0, 16) | reverted | n07 |  | 0.990 |
| 484 | ('hc_down', 0, 128) | reverted | n07 |  | 1.004 |
| 485 | ('hc_up', 0, 5) | trial | n07 | nt | 0.990 |
| 486 | ('hc_up', 0, 8) | reverted | n07 |  | 0.995 |
| 487 | ('hc_up', 0, 64) | trial | n07 | d6+bpre | 0.982 |
| 488 | ('hc_up', 0, 128) | reverted | n07 |  | 1.000 |
| 489 | ('router', 0, 5) | trial | n07 | nt+map | 0.969 |
| 490 | ('router', 0, 32) | trial | n07 | bpre | 0.977 |
| 491 | ('router', 0, 64) | trial | n07 | xpad | 0.986 |
| 492 | ('router', 0, 128) | trial | n07 | nosb+map+bpre | 0.986 |
| 493 | ('gdn_in', 2, 5) | reverted | n07 |  | 0.988 |
| 494 | ('gdn_in', 1, 8) | reverted | n07 |  | 0.965 |
| 495 | ('qsa_in', 2, 5) | reverted | n07 |  | 0.998 |
| 496 | ('qsa_in', 1, 8) | reverted | n07 |  | 0.976 |
| 497 | ('qsa_in', 1, 32) | trial | n07 | nt+map+xjunk | 0.973 |
| 498 | ('head', 2, 8) | reverted | n07 |  | 0.972 |
| 499 | ('head', 2, 32) | trial | n07 | w4+map | 0.927 |
| 500 | ('head', 2, 128) | confirmed | n07 |  | 0.945 |

Pending at the freeze (one-node WINs, not in the table): ('gdn_out', 2, 16) (dall 0.986 on n07), ('gdn_out', 2, 32) (mt*2+w4+nosb 0.961 on n07), ('hc_up', 0, 5) (nt 0.990 on n07), ('hc_up', 0, 64) (d6+bpre 0.982 on n07), ('head', 2, 32) (w4+map 0.927 on n07), ('qsa_in', 1, 32) (nt+map+xjunk 0.973 on n07), ('router', 0, 128) (nosb+map+bpre 0.986 on n07), ('router', 0, 32) (bpre 0.977 on n07), ('router', 0, 5) (nt+map 0.969 on n07), ('router', 0, 64) (xpad 0.986 on n07)

### 4.2 Lever ledger (every lookback round: each lever solo on the cell's pick of that round; phase ratio to the pick)

| lever | measurements | median | min | max | in WIN combinations |
|---|---|---|---|---|---|
| xpad | 512 | 0.998 | 0.941 | 1.039 | 45 |
| xjunk | 590 | 1.000 | 0.947 | 1.260 | 26 |
| fg32 | 15 | 1.001 | 0.999 | 1.023 | 3 |
| bpre | 767 | 1.001 | 0.941 | 1.074 | 51 |
| fg16 | 74 | 1.002 | 0.968 | 1.045 | 3 |
| xa2 | 262 | 1.003 | 0.924 | 1.108 | 46 |
| map | 1005 | 1.006 | 0.921 | 1.411 | 90 |
| nt | 1005 | 1.011 | 0.932 | 1.141 | 68 |
| d3 | 812 | 1.012 | 0.940 | 1.116 | 34 |
| d2 | 767 | 1.012 | 0.965 | 1.121 | 28 |
| d6 | 875 | 1.032 | 0.949 | 1.262 | 22 |
| w4 | 343 | 1.032 | 0.927 | 1.497 | 74 |
| nosb | 1005 | 1.041 | 0.949 | 1.257 | 58 |
| noxu | 1005 | 1.048 | 0.951 | 1.561 | 40 |
| dall | 544 | 1.051 | 0.955 | 1.281 | 13 |
| w8 | 283 | 1.090 | 1.003 | 1.337 | 0 |
| s/2 | 93 | 1.102 | 0.893 | 1.289 | 19 |
| xsrc | 889 | 1.133 | 0.905 | 3.308 | 18 |
| m32 | 195 | 1.169 | 1.021 | 1.554 | 0 |
| mt*2 | 681 | 1.170 | 0.974 | 1.542 | 16 |
| mt/2 | 589 | 1.179 | 0.896 | 1.450 | 16 |
| kw*2 | 439 | 1.211 | 0.940 | 1.773 | 7 |
| tb/2 | 346 | 1.226 | 0.808 | 1.790 | 21 |
| kw/2 | 545 | 1.260 | 0.924 | 1.694 | 37 |
| tb*2 | 257 | 1.292 | 1.013 | 1.669 | 0 |
| spread | 111 | 1.695 | 1.187 | 2.448 | 0 |
| bar | 1005 | 1.746 | 1.087 | 2.757 | 0 |
| s*2 | 542 | 2.210 | 0.980 | 4.842 | 0 |

Lever names: `xpad` LDS row pad 32, `xjunk` branch-free staging toggle, `bpre` B operands read ahead, `fg16` / `fg32` fold loads in flight, `xa2` two activation chunks in flight, `map` rr <-> xcd, `nt` non-temporal weight loads, `d2` / `d3` / `d6` / `dall` weight prefetch depth, `w4` / `w8` waves per CTA, `nosb` no sched_barrier fences, `noxu` no cross-unit prefetch, `s/2` / `s*2` K splits, `xsrc` LDS <-> global B operands, `mt*2` / `mt/2` weight tiles per wave, `kw*2` / `kw/2` K groups per CTA, `tb/2` / `tb*2` token tiles per unit, `spread` fold=spread, `m32` the 32x32x64 scaled MFMA (32-row weight tiles, half the weight tiles per wave), `bar` the P-L3 barrier instead of persist.

### 4.3 Combination search per round

| round | combination mode | combinations timed | WIN finalists (one node) |
|---|---|---|---|
| rL1a | top6 | 438 | 18 |
| rL2b | top6 | 1886 | 53 |
| rL3a | top6 | 1892 | 30 |
| rL4b | top6 | 1860 | 17 |
| rL5a | top6 | 1823 | 15 |
| rL6b | top6 | 1812 | 18 |
| rL7a | top6 | 1789 | 18 |
| rL8b | top6 | 1756 | 24 |
| rL9a | all2 | 7919 | 32 |
| rL10d | all2x | 11677 | 38 |
| rL11a | all2x | 11702 | 31 |
| rL12b | all2x | 11560 | 17 |
| rL13a | all2x | 11494 | 27 |
| rL14b | all2x | 11534 | 14 |
| rL15a | all2x | 11535 | 17 |
| rL16b | all2x | 11608 | 13 |

### 4.4 WIN finalists made only of solo losers (every lever of the set lost solo on that base; PLAN 13)

| round | cell | lever set | ratio to the pick | solo ratios |
|---|---|---|---|---|
| rL1a | hc_up_tpx_r8 | nt+d3 | 0.987 | 1.001 / 1.002 |
| rL3a | head_tp2_r64 | noxu+nosb+xjunk | 0.979 | 1.000 / 1.003 / 1.004 |
| rL3a | head_tp1_r32 | nosb+w4 | 0.984 | 1.001 / 1.004 |
| rL4b | hc_down_tpx_r5 | nt+nosb+xa2 | 0.985 | 1.002 / 1.004 / 1.010 |
| rL5a | gdn_out_tp2_r5 | map+bpre | 0.954 | 1.041 / 1.049 |
| rL5a | hc_down_tpx_r8 | noxu+nt+xpad | 0.989 | 1.003 / 1.007 / 1.015 |
| rL5a | hc_down_tpx_r32 | d2+noxu | 0.965 | 1.003 / 1.004 |
| rL6b | hc_down_tpx_r16 | nt+dall | 0.989 | 1.003 / 1.009 |
| rL6b | head_tp1_r128 | nosb+w4+map | 0.942 | 1.001 / 1.001 / 1.002 |
| rL6b | head_tp1_r128 | nosb+w4+xjunk | 0.950 | 1.001 / 1.001 / 1.004 |
| rL7a | hc_down_tpx_r32 | noxu+d2 | 0.969 | 1.004 / 1.004 |
| rL9a | gdn_out_tp2_r128 | kw/2+tb/2 | 0.909 | 1.113 / 1.245 |
| rL9a | gdn_out_tp2_r128 | kw/2+w4 | 0.936 | 1.113 / 1.369 |
| rL9a | gdn_out_tp1_r128 | tb/2+mt*2 | 0.915 | 1.173 / 1.186 |
| rL9a | gdn_out_tp1_r128 | tb/2+kw/2 | 0.956 | 1.173 / 1.282 |
| rL9a | hc_down_tpx_r64 | mt*2+kw*2 | 0.980 | 1.064 / 1.246 |
| rL9a | hc_up_tpx_r8 | nosb+xsrc | 0.986 | 1.089 / 1.161 |
| rL9a | gdn_in_tp2_r32 | w4+kw/2 | 0.953 | 1.159 / 1.275 |
| rL9a | gdn_in_tp2_r64 | mt*2+w4 | 0.939 | 1.207 / 1.219 |
| rL9a | gdn_in_tp2_r64 | kw/2+tb/2 | 0.980 | 1.095 / 1.173 |
| rL9a | gdn_in_tp1_r32 | w4+kw/2 | 0.957 | 1.122 / 1.271 |
| rL9a | qsa_in_tp2_r64 | tb/2+kw/2 | 0.904 | 1.111 / 1.220 |
| rL9a | qsa_in_tp2_r64 | tb/2+mt*2 | 0.957 | 1.111 / 1.221 |
| rL9a | qsa_in_tp2_r128 | kw/2+tb/2 | 0.888 | 1.095 / 1.195 |
| rL9a | qsa_in_tp2_r128 | tb/2+mt*2 | 0.933 | 1.195 / 1.232 |
| rL10d | gdn_out_tp2_r32 | kw/2+tb/2+xa2 | 0.943 | 1.141 / 1.152 / 1.073 |
| rL10d | hc_down_tpx_r64 | mt*2+kw*2+nosb | 0.956 | 1.049 / 1.265 / 1.021 |
| rL10d | hc_down_tpx_r64 | mt*2+kw*2+s/2 | 0.972 | 1.049 / 1.265 / 1.062 |
| rL10d | hc_up_tpx_r32 | xa2+nosb | 0.972 | 1.006 / 1.019 |
| rL10d | hc_up_tpx_r32 | xa2+nosb+xjunk | 0.974 | 1.006 / 1.019 / 1.013 |
| rL10d | router_tpx_r128 | map+nosb+d3 | 0.988 | 1.011 / 1.015 / 1.007 |
| rL10d | gdn_in_tp1_r5 | w4+kw/2+bpre | 0.960 | 1.081 / 1.449 / 1.010 |
| rL10d | gdn_in_tp1_r128 | xa2+w4+nt | 0.976 | 1.021 / 1.079 / 1.001 |
| rL10d | qsa_in_tp2_r32 | kw/2+tb/2+d6 | 0.959 | 1.105 / 1.133 / 1.045 |
| rL10d | qsa_in_tp2_r32 | kw/2+tb/2+d2 | 0.967 | 1.105 / 1.133 / 1.031 |
| rL10d | qsa_in_tp2_r128 | kw/2+tb/2+w4 | 0.837 | 1.105 / 1.190 / 1.235 |
| rL10d | qsa_in_tp1_r64 | tb/2+kw/2+d2 | 0.954 | 1.167 / 1.196 / 1.005 |
| rL11a | gdn_out_tp2_r64 | kw/2+mt/2+xa2 | 0.969 | 1.156 / 1.329 / 1.059 |
| rL11a | hc_down_tpx_r5 | nt+d2+noxu | 0.968 | 1.016 / 1.019 / 1.020 |
| rL11a | hc_down_tpx_r32 | d3+noxu+fg16 | 0.992 | 1.005 / 1.008 / 1.026 |
| rL11a | gdn_in_tp1_r5 | mt/2+kw/2 | 0.944 | 1.035 / 1.300 |
| rL11a | gdn_in_tp1_r8 | w4+kw/2+d6 | 0.955 | 1.080 / 1.461 / 1.066 |
| rL11a | gdn_in_tp1_r8 | w4+kw/2+d2 | 0.938 | 1.080 / 1.461 / 1.014 |
| rL11a | gdn_in_tp1_r16 | w4+kw/2+d6 | 0.962 | 1.155 / 1.475 / 1.085 |
| rL11a | gdn_in_tp1_r128 | xa2+w4+noxu | 0.977 | 1.006 / 1.070 / 1.014 |
| rL11a | gdn_in_tp1_r128 | xa2+w4+nt | 0.966 | 1.006 / 1.070 / 1.002 |
| rL11a | head_tp2_r128 | noxu+nosb+w4 | 0.953 | 1.011 / 1.021 / 1.015 |
| rL12b | hc_down_tpx_r16 | dall+bpre | 0.979 | 1.013 / 1.024 |
| rL12b | hc_up_tpx_r64 | xsrc+nosb+map | 0.980 | 1.016 / 1.111 / 1.001 |
| rL12b | gdn_in_tp1_r16 | w4+kw/2+d2 | 0.964 | 1.053 / 1.411 / 1.010 |
| rL13a | gdn_in_tp1_r8 | w4+kw/2 | 0.951 | 1.015 / 1.422 |
| rL13a | gdn_in_tp1_r16 | mt/2+kw/2+d3 | 0.950 | 1.070 / 1.300 / 1.025 |
| rL13a | qsa_in_tp1_r5 | mt/2+kw/2+d2 | 0.965 | 1.094 / 1.338 / 1.013 |
| rL14b | gdn_out_tp2_r32 | mt*2+w4+nosb | 0.929 | 1.138 / 1.209 / 1.081 |
| rL14b | gdn_out_tp2_r128 | w4+mt*2+d6 | 0.968 | 1.273 / 1.282 / 1.025 |
| rL14b | hc_up_tpx_r16 | nosb+xsrc+map | 0.985 | 1.092 / 1.153 / 1.011 |
| rL14b | router_tpx_r128 | kw/2+tb/2+bpre | 0.933 | 1.079 / 1.213 / 1.051 |
| rL14b | gdn_in_tp1_r16 | w4+kw/2+d6 | 0.957 | 1.040 / 1.401 / 1.053 |
| rL14b | qsa_in_tp1_r32 | w4+kw/2+d6 | 0.967 | 1.214 / 1.231 / 1.073 |
| rL15a | hc_down_tpx_r16 | nt+dall+bpre | 0.981 | 1.012 / 1.017 / 1.024 |
| rL15a | hc_up_tpx_r8 | nosb+xsrc+mt*2 | 0.979 | 1.115 / 1.173 / 1.023 |
| rL15a | gdn_in_tp1_r8 | w4+kw/2+d6 | 0.925 | 1.013 / 1.417 / 1.057 |
| rL15a | head_tp2_r128 | bpre+w4+nosb | 0.947 | 1.007 / 1.008 / 1.012 |
| rL16b | gdn_out_tp2_r32 | mt*2+w4+nosb | 0.961 | 1.191 / 1.223 / 1.134 |
| rL16b | hc_up_tpx_r64 | d6+bpre | 0.982 | 1.009 / 1.011 |
| rL16b | router_tpx_r128 | nosb+map+bpre | 0.986 | 1.026 / 1.028 / 1.007 |

## 5. Plan-of-record screens (four runs per projection group, two per node; each run's best confirmed arm vs P-L3's arm and FlyTape's plan replicas; the screen-2 lever ledger per run is in the private raw JSON)

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

