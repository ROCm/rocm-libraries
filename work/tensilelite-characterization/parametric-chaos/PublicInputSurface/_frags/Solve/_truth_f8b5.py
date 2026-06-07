import sys
try:
    from Tensile.Configuration import Parameter
except Exception as e:
    print("IMPORT ERR:", repr(e))
    sys.exit(0)
p = Parameter(name="x", initialValue=1)
domain = [
    ("Parameter(name=x, initialValue=1)", p),
    ("42", 42),
    ("3.14", 3.14),
    ("hello-str", "hello"),
]
print("Parameter class:", Parameter)
for label, val in domain:
    print("isinstance(%-35s, Parameter) = %s" % (label, isinstance(val, Parameter)))
