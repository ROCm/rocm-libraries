# Toolchain/Validators.py — characterization target

Pins the toolchain component validators: `supported*` predicates,
`_supportedComponent`, `_exeExists`, `_posixSearchPaths`, `_windowsSearchPaths`,
`_windowsLatestRocmBin`, `_windowsWithExtensions`, `_validateExecutable`,
`validateToolchain`, and `ToolchainDefaults`.

Coverage: 96 stmts, 0 missed → 100% line (99.24% blended; one residual branch
`75->78` = ROCm dir exists but has no `X.Y` subdirs, a Windows-only corner).

Windows-only paths are reached on the Linux host by calling the helpers
directly (they do not all gate on `os.name`) and by monkeypatching
`V.os.name='nt'` + `PATHEXT` for the extension-expansion branches, and
`DEFAULT_ROCM_BIN_PATH_WINDOWS` to a tmp dir for the latest-bin append.
