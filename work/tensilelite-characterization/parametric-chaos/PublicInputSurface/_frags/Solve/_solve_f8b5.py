from z3 import Datatype, Solver, sat, Function, BoolSort, Const, Or, Not

Lhs = Datatype("Lhs")
labels = ["v_param", "v_42", "v_314", "v_hello"]
for l in labels:
    Lhs.declare(l)
Lhs = Lhs.create()
ctors = {l: getattr(Lhs, l) for l in labels}
TAG = {"v_param": True, "v_42": False, "v_314": False, "v_hello": False}

is_parameter = Function("is_parameter", Lhs, BoolSort())
lhs = Const("lhs", Lhs)
base = [Or([lhs == ctors[l] for l in labels])]
for l in labels:
    base.append(is_parameter(ctors[l]) == TAG[l])
pred = is_parameter(lhs)

def decode(m):
    lv = str(m.eval(lhs, model_completion=True))
    return lv

s = Solver(); [s.add(c) for c in base]; s.add(pred)
rt = s.check(); mt = s.model() if rt == sat else None
s2 = Solver(); [s2.add(c) for c in base]; s2.add(Not(pred))
rf = s2.check(); mf = s2.model() if rf == sat else None
print("TRUE_CHECK", rt, "SEL", decode(mt) if mt else None)
print("FALSE_CHECK", rf, "SEL", decode(mf) if mf else None)
