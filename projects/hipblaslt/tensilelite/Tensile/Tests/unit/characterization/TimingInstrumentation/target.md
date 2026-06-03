# Characterization target — `Tensile/Common/TimingInstrumentation.py`

Part of the master-plan remaining-module sweep. **Before 76.2% → after 100.00%
line** (21 stmts, 0 miss; 96% blended). Drives `timing_context` on/off branches
(gated by `globalParameters["TimingInstrumentation"]`, saved/restored). Residual:
1 partial branch (33->41 — the module-import `if not _timing_logger.handlers`
guard, already taken at import; line coverage is 100%).
