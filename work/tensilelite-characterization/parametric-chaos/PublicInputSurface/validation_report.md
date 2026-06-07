# Parametric-Chaos Run-1 Validation Report

Generated: 2026-06-07  
Branch set: 20 unique branch IDs  
Validation methods used: z3 (bounded SAT / UNSAT), CrossHair (bounded symbolic), pytest pass-check (real entry-point)

---

## Per-unit table

| branch_id (12) | file:line | classification | solver_status | confirmed | reified? |
|---|---|---|---|---|---|
| `01e8ac7f3712` | Tensile/Tensile.py:529 | fully-static | SAT | yes | no |
| `05506103d267` | Tensile/Configuration.py:630 | runtime-dependent | SAT | yes | yes |
| `2075748886b1` | Tensile/Configuration.py:730 | runtime-dependent | SAT | yes | yes |
| `26bfafbb23ef` | Tensile/Configuration.py:692 | runtime-dependent | UNKNOWN | yes | no |
| `26f1acfe1ff9` | Tensile/Tensile.py:603 | fully-static | SAT | yes | yes |
| `2c7170bfd056` | Tensile/Tensile.py:25 | fully-static | SAT | yes | yes |
| `3c77ffccaeef` | Tensile/Configuration.py:579 | runtime-dependent | UNKNOWN | yes | no |
| `4914224d6e01` | Tensile/Common/GlobalParameters.py:660 | fully-static | SAT | yes | yes |
| `5e52e9474f01` | Tensile/Configuration.py:929 | runtime-dependent | SAT | yes | yes |
| `765305e2fbcf` | Tensile/Tensile.py:239 | fully-static | SAT | yes | yes |
| `766aca336236` | Tensile/Configuration.py:218 | runtime-dependent | SAT | yes | yes |
| `8226b3bb80ef` | Tensile/Configuration.py:673 | runtime-dependent | SAT | yes | yes |
| `8f7c4911b799` | Tensile/Tensile.py:536 | runtime-dependent | UNKNOWN | no | no |
| `aa18a787b08b` | Tensile/Tensile.py:409 | fully-static | SAT | yes | yes |
| `b87e16eec6ca` | Tensile/Configuration.py:230 | runtime-dependent | UNKNOWN | no | no |
| `c63babfc10d3` | Tensile/Tensile.py:534 | fully-static | SAT | yes | yes |
| `cab4f49fe2f4` | Tensile/Configuration.py:534 | runtime-dependent | SAT | yes | no |
| `d8f43265b665` | Tensile/Tensile.py:526 | fully-static | SAT | yes | yes |
| `f6f7dc557d11` | Tensile/Common/GlobalParameters.py:659 | fully-static | SAT | yes | yes |
| `f8b5af6a1a52` | Tensile/Configuration.py:224 | runtime-dependent | UNKNOWN | yes | no |

---

## Summary counts

| metric | count |
|---|---|
| Total branches inventoried | 20 |
| SAT | 15 |
| UNSAT | 0 |
| UNKNOWN | 5 |
| Witnesses confirmed | 18 |
| Tests reified | 13 |

---

## UNKNOWN branches — explicit statements

**`8f7c4911b799` — Tensile/Tensile.py:536 (`not os.path.exists(restoreLogPath)`)**  
Solver status: UNKNOWN. Confirmed: no.  
Rationale: predicate truth depends on whether the path named by `--restore-from-log` exists in the live filesystem at runtime. os.path.exists is a genuine filesystem probe; z3 cannot enumerate all possible filesystem states. The two branch outcomes are both structurally reachable (confirmed code-path analysis), but no pytest witness was produced because fabricating a sentinel file path would exercise a runtime assumption, not a solver-confirmed fact.

**`b87e16eec6ca` — Tensile/Configuration.py:230 (`isinstance(rhs, Parameter)`)**  
Solver status: UNKNOWN. Confirmed: no (runtime-dependent).  
Rationale: the isinstance check on a live Python object type cannot be encoded as a simple SMT formula. Whether rhs is a Parameter depends on the call stack's construction of objects, not on any CLI or YAML input directly. z3 confirmed reachability of both branches via explicit witness construction, but the confirm flag is false because the Verify agent noted residual uncertainty about all call paths.

**`3c77ffccaeef` — Tensile/Configuration.py:579 (`isinstance(op, str)`)**  
Solver status: UNKNOWN. Confirmed: yes (witnesses verified in-container against real createUnaryOp).  
Rationale: z3 cannot prove isinstance; CrossHair ran but produced a tautological result for the abstracted helper. Both branches (str op vs callable op) were exercised against the real code in-container and behaved as claimed. Downgraded from sat to unknown by the Verify agent per rigor rules.

**`26bfafbb23ef` — Tensile/Configuration.py:692 (`nodeType == "Assign"`)**  
Solver status: UNKNOWN. Confirmed: yes.  
Rationale: nodeType = type(node).__name__ is structurally determined by the AST produced from expressionStr. The Assign branch is reachable only from test code (no production call path supplies assignment statements). Witnesses confirmed in-container. UNKNOWN because no solver could enumerate all call paths.

**`f8b5af6a1a52` — Tensile/Configuration.py:224 (`isinstance(lhs, Parameter)`)**  
Solver status: UNKNOWN. Confirmed: yes.  
Rationale: same isinstance constraint as `b87e16eec6ca`. Both branch outcomes confirmed by running real code in-container with explicit Parameter and non-Parameter lhs values.

---

## Runtime-dependent branches (never silently asserted)

The following branches have predicate truth values that depend on runtime state (filesystem, OS invocation mode, live Python object types) and are **not** captured by any CLI/YAML/global-parameter input:

- `8f7c4911b799` Tensile.py:536 — filesystem probe (os.path.exists)
- `2c7170bfd056` Tensile.py:25 — Python interpreter invocation mode (`__name__ == "__main__"`)
- `8f7c4911b799` Tensile.py:536 — CWD (os.getcwd() for relative path resolution)
- `3c77ffccaeef`, `b87e16eec6ca`, `f8b5af6a1a52` — Python runtime isinstance on live objects

The branch `2c7170bfd056` (Tensile.py:25, `__name__ == "__main__"`) is classified fully-static and SAT because the witness is trivially: run the file as a script (True) vs import it (False). Both outcomes are confirmed.

---

## Validation methods used

| method | applied to |
|---|---|
| z3 (bounded SAT, z3-solver 4.16.0) | all 20 branches in Solve phase |
| CrossHair 0.0.106 (bounded symbolic) | 01e8ac7f3712, and branches where a pure helper was synthesizable |
| argparse reconstruction (in-container) | Tensile.py CLI branches (01e8ac7f, d8f43265, c63babfc, 765305e2) |
| pytest pass-check (no coverage) | 13 reified branches |
| Code reading + in-container execution | all 20 branches in Verify phase |
