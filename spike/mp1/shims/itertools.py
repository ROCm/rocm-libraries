# Minimal `itertools` shim for MicroPython — ck_dsl uses product + accumulate.


def product(*iterables, repeat=1):
    pools = [tuple(p) for p in iterables] * repeat
    result = [[]]
    for pool in pools:
        result = [x + [y] for x in result for y in pool]
    for prod in result:
        yield tuple(prod)


def accumulate(iterable, func=None, *, initial=None):
    it = iter(iterable)
    total = initial
    if total is None:
        try:
            total = next(it)
        except StopIteration:
            return
    yield total
    for element in it:
        total = func(total, element) if func is not None else total + element
        yield total


def chain(*iterables):
    for it in iterables:
        for x in it:
            yield x


def repeat(obj, times=None):
    if times is None:
        while True:
            yield obj
    else:
        for _ in range(times):
            yield obj


def count(start=0, step=1):
    n = start
    while True:
        yield n
        n += step


def islice(iterable, *args):
    s = slice(*args)
    start = s.start or 0
    stop = s.stop
    step = s.step or 1
    for i, x in enumerate(iterable):
        if i < start:
            continue
        if stop is not None and i >= stop:
            return
        if (i - start) % step == 0:
            yield x
