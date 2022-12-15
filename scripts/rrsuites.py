from rrperf import GEMMRun, CodeGenRun


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
    yield GEMMRun(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        type_A="half",
        type_B="half",
        type_C="half",
        type_D="half",
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )


def sgemm():
    yield GEMMRun(M=3072, N=4096, K=4096, mac_m=64, mac_n=64, mac_k=64)
    yield GEMMRun(M=3072, N=4096, K=4096, mac_m=128, mac_n=64, mac_k=16)


def hgemm():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        type_A="half",
        type_B="half",
        type_C="half",
        type_D="half",
    )
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=8,
        type_A="half",
        type_B="half",
        type_C="half",
        type_D="half",
    )


def codegen():
    yield CodeGenRun(instCount=40000, instructions="comments")
    yield CodeGenRun(instCount=40000, instructions="simple_mfma")
    yield CodeGenRun(instCount=40000, instructions="complex_mfma_with_coop")


def all():
    yield from sgemm()
    yield from hgemm()
    yield from codegen()
