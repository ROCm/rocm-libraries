import json, itertools
from z3 import Bool, Not, Or, Solver, sat, is_true

exists = Bool("exists")   # os.path.exists(resultsFileName)
force  = Bool("force")    # globalParameters["ForceRedoBenchmarkProblems"]
pred = Or(Not(exists), force)

def model_for(want_true):
    s = Solver()
    s.add(pred if want_true else Not(pred))
    if s.check() == sat:
        m = s.model()
        return {"exists": bool(is_true(m.eval(exists, model_completion=True))),
                "force":  bool(is_true(m.eval(force,  model_completion=True)))}
    return None

out = {"true_model": model_for(True),
       "false_model": model_for(False),
       "table": [{"exists": e, "force": f, "pred": (not e) or f}
                 for e,f in itertools.product([False,True],[False,True])]}
print(json.dumps(out, indent=2))
