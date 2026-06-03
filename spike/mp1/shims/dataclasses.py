# Declaration-order-preserving dataclasses shim for MicroPython.
#
# MicroPython does not retain class __annotations__ and its dicts are unordered,
# so fields cannot be discovered from bare annotations nor ordered by __dict__.
# Convention required of ck_dsl: every dataclass field is declared with an
# explicit `field(...)` (or a `field(default=...)`) value, e.g.
#     op: str = field()
#     dtype: str = field(default="f16")
# Each field() call carries a global creation counter, so we recover *declaration
# order* (unlike udataclasses, which sorts alphabetically and is keyword-only).
#
# Supported: positional + keyword __init__, required fields, default,
# default_factory, frozen (eq/hash/immutability), eq, repr, replace(), fields(),
# __post_init__, single-level inheritance of fields. No exec/compile used.


class _Missing:
    def __repr__(self):
        return "MISSING"


MISSING = _Missing()

_counter = [0]


class FrozenInstanceError(AttributeError):
    pass


class Field:
    def __init__(self, default=MISSING, default_factory=MISSING):
        self.default = default
        self.default_factory = default_factory
        self.name = None
        self.order = _counter[0]
        _counter[0] += 1


def field(default=MISSING, default_factory=MISSING):
    return Field(default=default, default_factory=default_factory)


_FIELDS = "__dataclass_fields__"
# Ordered tuple of fields — MicroPython dict .values() is unordered, so the
# declaration order must be stored explicitly rather than read back from _FIELDS.
_FIELDS_ORDER = "__dataclass_fields_order__"


def _collect(cls):
    found = {}
    # Inherit base fields first (preserve their order via their stored order key).
    for base in getattr(cls, "__bases__", ()):  # MicroPython exposes __bases__
        bf = getattr(base, _FIELDS, None)
        if bf:
            for n, f in bf.items():
                found[n] = f
    # Own fields: Field instances sitting in the class dict.
    for name, val in list(cls.__dict__.items()):
        if isinstance(val, Field):
            val.name = name
            found[name] = val
    ordered = sorted(found.values(), key=lambda f: f.order)
    return ordered


def _make_init(fields, frozen):
    def __init__(self, *args, **kwargs):
        if len(args) > len(fields):
            raise TypeError("__init__() takes at most %d positional args" % len(fields))
        values = {}
        for i, a in enumerate(args):
            values[fields[i].name] = a
        for k, v in kwargs.items():
            if k in values:
                raise TypeError("got multiple values for argument '%s'" % k)
            values[k] = v
        for f in fields:
            if f.name in values:
                val = values[f.name]
            elif f.default is not MISSING:
                val = f.default
            elif f.default_factory is not MISSING:
                val = f.default_factory()
            else:
                raise TypeError("missing required argument: '%s'" % f.name)
            object.__setattr__(self, f.name, val)
        post = getattr(self, "__post_init__", None)
        if post is not None:
            post()

    return __init__


def _make_eq(fields):
    names = [f.name for f in fields]

    def __eq__(self, other):
        if type(self) is not type(other):
            return NotImplemented
        for n in names:
            if getattr(self, n) != getattr(other, n):
                return False
        return True

    return __eq__


def _make_hash(fields):
    names = [f.name for f in fields]

    def __hash__(self):
        return hash(tuple(getattr(self, n) for n in names))

    return __hash__


def _make_repr(cls, fields):
    names = [f.name for f in fields]

    def __repr__(self):
        parts = ", ".join("%s=%r" % (n, getattr(self, n)) for n in names)
        return "%s(%s)" % (cls.__name__, parts)

    return __repr__


def _frozen_setattr(self, name, value):
    raise FrozenInstanceError("cannot assign to field '%s'" % name)


def _frozen_delattr(self, name):
    raise FrozenInstanceError("cannot delete field '%s'" % name)


def _process(cls, frozen, eq, order):
    fields = _collect(cls)
    setattr(cls, _FIELDS, {f.name: f for f in fields})
    setattr(cls, _FIELDS_ORDER, tuple(fields))
    # Clear the Field markers off the class so attribute access returns instance
    # values (or raises) rather than the Field object.
    for f in fields:
        if f.name in cls.__dict__:
            try:
                delattr(cls, f.name)
            except Exception:
                pass
    cls.__init__ = _make_init(fields, frozen)
    cls.__repr__ = _make_repr(cls, fields)
    if eq:
        cls.__eq__ = _make_eq(fields)
        if frozen:
            cls.__hash__ = _make_hash(fields)
    if frozen:
        cls.__setattr__ = _frozen_setattr
        cls.__delattr__ = _frozen_delattr
    return cls


def dataclass(cls=None, *, frozen=False, eq=True, order=False, **_ignored):
    def wrap(c):
        return _process(c, frozen, eq, order)

    if cls is None:
        return wrap
    return wrap(cls)


def fields(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return getattr(cls, _FIELDS_ORDER)


def is_dataclass(obj):
    cls = obj if isinstance(obj, type) else type(obj)
    return hasattr(cls, _FIELDS)


def replace(obj, **changes):
    cls = type(obj)
    kwargs = {}
    for f in getattr(cls, _FIELDS_ORDER):
        kwargs[f.name] = changes[f.name] if f.name in changes else getattr(obj, f.name)
    return cls(**kwargs)


def asdict(obj):
    return {f.name: getattr(obj, f.name) for f in getattr(type(obj), _FIELDS_ORDER)}


def astuple(obj):
    return tuple(getattr(obj, f.name) for f in getattr(type(obj), _FIELDS_ORDER))
