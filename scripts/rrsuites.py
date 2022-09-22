from rrperf import GEMMRun


def unit():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )


def sgemm():
    yield GEMMRun(M=3072, N=4096, K=4096, mac_m=64, mac_n=64, mac_k=64)
    yield GEMMRun(M=3072, N=4096, K=4096, mac_m=128, mac_n=64, mac_k=16)


def all():
    yield from sgemm()
