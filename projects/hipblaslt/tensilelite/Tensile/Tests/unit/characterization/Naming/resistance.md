# Resisting lines — `Tensile/SolutionStructs/Naming.py`

The suite reaches **99.17% line** standalone (120 stmts, 1 miss). The single
residual line is provably unreachable; one notable **latent bug** was
characterized (pinned as current behaviour) rather than worked around. New file
in the per-target dir per the add-only rule.

## Unreachable LINE (1) — counted as Miss, cannot be hit

| Line | Code | Why unreachable |
|---|---|---|
| 141 | `raise Exception(f"Parameter {key}={value} is new object type ({type(value)})")` in `getParameterValueAbbreviation` | The function first returns for non-composite values (L132-133), then handles `tuple` / `list` / `dict` explicitly (L134-139). The `else: raise` is reached only when `value` is a composite (`isinstance(value, (dict, list, tuple))` is True) **but** matches none of those three — impossible. Dead defensive code. |

## Characterized BUG (pinned, not worked around)

| Where | Behaviour | Test |
|---|---|---|
| `_getName`, L154-155 then L160 (via `getKernelNameMin` / any `ignoreInternalArgs=True` caller) | With `splitGSU=True` and `GlobalSplitU > 1` or `== -1`, L155 rewrites `state["GlobalSplitU"]` to the string `"M"`; L160 then evaluates `state["GlobalSplitU"] > 0`, i.e. `"M" > 0` → **`TypeError: '>' not supported between 'str' and 'int'`**. | `test_kernel_name_min_split_gsu_typeerror[gsu4]` / `[gsu_neg1]` assert the `TypeError` (current behaviour). `GSU==1` (not rewritten) works and is snapshotted by `test_kernel_name_min_split_gsu_unit`. |

This is a genuine latent defect in the module, not a test artifact: any
`ignoreInternalArgs=True` name request (`getKernelNameMin`, and
`getKernelFileBase`/`shortenFileBase` which call it) with `splitGSU=True` on a
kernel whose `GlobalSplitU > 1` (or `-1`) raises. It is **characterized**
(pinned via `pytest.raises`) so a future fix will surface as an intentional
snapshot/expectation change. Fixing it is out of scope here (add-only;
characterization pins current behaviour).

## Branch note

`118->exit` (the `getPrimitiveParameterValueAbbreviation` fall-through to
implicit `None` for an unhandled type) is covered by
`test_primitive_abbrev_unhandled_type_returns_none` (passing `None`).

## Determinism technique (not a gap)

- Every output is a pure function of the input `state`; names are snapshotted
  directly. The solution `state` is built by the `conftest.make_state` factory
  (a real `ProblemType` + the `GlobalSplitU` / internal-args / tile keys the
  builders read), so snapshots are stable.
- `_getName` and `getKeyNoInternalArgs` **mutate then restore**
  `state["GlobalSplitU"]` and `state["ProblemType"]["GroupedGemm"]`;
  dedicated tests (`test_name_does_not_mutate_state`,
  `test_key_no_internal_args_restores_state`) pin that the restore is exact.
- `shortenFileBase`'s long path is a deterministic sha256+base64 of the name
  tail, so the shortened string is snapshot-stable.
