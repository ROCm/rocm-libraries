# Minimal `functools` shim for MicroPython (lru_cache + the common helpers).


def lru_cache(maxsize=128, typed=False):
    def deco(fn):
        cache = {}

        def wrapper(*args, **kwargs):
            key = args if not kwargs else (args, tuple(sorted(kwargs.items())))
            if key in cache:
                return cache[key]
            r = fn(*args, **kwargs)
            cache[key] = r
            return r

        wrapper.cache_clear = cache.clear
        wrapper.__wrapped__ = fn
        return wrapper

    # Bare @lru_cache (no parens): maxsize is actually the function.
    if callable(maxsize):
        fn = maxsize
        maxsize = 128
        return deco(fn)
    return deco


def wraps(wrapped, *args, **kwargs):
    def deco(fn):
        for attr in ("__name__", "__doc__", "__module__"):
            try:
                setattr(fn, attr, getattr(wrapped, attr))
            except (AttributeError, TypeError):
                pass
        return fn

    return deco


def reduce(function, iterable, *initial):
    it = iter(iterable)
    if initial:
        acc = initial[0]
    else:
        acc = next(it)
    for x in it:
        acc = function(acc, x)
    return acc


class partial:
    def __init__(self, func, *args, **kwargs):
        self.func = func
        self.args = args
        self.keywords = kwargs

    def __call__(self, *args, **kwargs):
        merged = dict(self.keywords)
        merged.update(kwargs)
        return self.func(*(self.args + args), **merged)
