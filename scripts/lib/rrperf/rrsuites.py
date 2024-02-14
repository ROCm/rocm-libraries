import pathlib
from rrperf.problems import GEMMRun, CodeGenRun, TensileRun

repo_dir = pathlib.Path(__file__).resolve().parent.parent.parent.parent

fp16 = {
    "type_A": "half",
    "type_B": "half",
    "type_C": "half",
    "type_D": "half",
}

fp32 = {
    "type_A": "float",
    "type_B": "float",
    "type_C": "float",
    "type_D": "float",
}

tensile_guidepost_HGEMM = {
    "M": 7680,
    "N": 8448,
    "K": 8448,  # matches the guidepost unit test
    "trans_B": "T",
}


def unit():
    yield GEMMRun(
        M=1024,
        N=1024,
        K=128,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        **fp32,
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
        **fp16,
        numWarmUp=1,
        numOuter=1,
        numInner=1,
    )


def sgemm():
    yield GEMMRun(
        M=3072,
        N=4096,
        K=4096,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        **fp32,
    )
    yield GEMMRun(
        M=3072,
        N=4096,
        K=4096,
        mac_m=128,
        mac_n=64,
        mac_k=16,
        **fp32,
    )


def hgemm_tensile_guidepost():
    yield GEMMRun(
        **tensile_guidepost_HGEMM,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        workgroup_size_x=128,
        workgroup_size_y=2,
        **fp16,
    )


def hgemm():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        **fp16,
    )
    # yield GEMMRun(
    #     M=7680,
    #     N=8448,
    #     K=8192,
    #     mac_m=128,
    #     mac_n=256,
    #     mac_k=16,
    #     workgroup_size_x=256,
    #     workgroup_size_y=1,
    #     **fp16,
    # )
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        **fp16,
    )
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        trans_A="N",
        trans_B="T",
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=4,
        **fp16,
    )
    yield from hgemm_tensile_guidepost()
    yield GEMMRun(
        **tensile_guidepost_HGEMM,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=256,
        workgroup_size_y=1,
        **fp16,
    )

    yield GEMMRun(
        **tensile_guidepost_HGEMM,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        **fp16,
    )

    yield GEMMRun(
        **tensile_guidepost_HGEMM,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=64,
        workgroup_size_y=4,
        **fp16,
    )

    yield GEMMRun(
        **tensile_guidepost_HGEMM,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )

    for sched in ["Priority", "Cooperative", "Sequential"]:
        yield GEMMRun(
            M=3840,
            N=4224,
            K=4224,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield GEMMRun(
            M=1024,
            N=50304,
            K=8192,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield GEMMRun(
            M=3840,
            N=4224,
            K=4224,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            betaInFma=False,
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield GEMMRun(
            M=1024,
            N=50304,
            K=8192,
            mac_m=128,
            mac_n=128,
            mac_k=32,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            betaInFma=False,
            visualize=False,
            scheduler=sched,
            **fp16,
        )

        yield GEMMRun(
            M=8448,
            N=8448,
            K=128,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        trans_A="T",
        trans_B="N",
        **fp16,
    )

    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        trans_A="T",
        trans_B="T",
        **fp16,
    )

    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=64,
        mac_n=64,
        mac_k=64,
        trans_A="N",
        trans_B="T",
        **fp16,
    )

    for sched in ["Priority", "Cooperative", "Sequential"]:
        yield GEMMRun(
            M=7680,
            N=8448,
            K=8192,
            mac_m=128,
            mac_n=256,
            mac_k=16,
            workgroup_size_x=256,
            workgroup_size_y=1,
            trans_A="N",
            trans_B="T",
            visualize=False,
            scheduler=sched,
            **fp16,
        )

    # yield GEMMRun(
    #     M=7680,
    #     N=8448,
    #     K=8192,
    #     mac_m=128,
    #     mac_n=256,
    #     mac_k=16,
    #     workgroup_size_x=256,
    #     workgroup_size_y=1,
    #     trans_A="T",
    #     trans_B="N",
    #     visualize=False,
    #     scheduler=sched,
    #     match_memory_access=True,
    #     **fp16,
    # )

    yield from visualizer()


def visualizer():
    yield GEMMRun(
        M=512,
        N=768,
        K=512,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        alpha=2.0,
        beta=0.5,
        workgroup_size_x=256,
        workgroup_size_y=1,
        trans_A="N",
        trans_B="T",
        storeLDS_D=False,
        visualize=True,
        prefetch=False,
        **fp16,
    )


def guidepost_1():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        visualize=False,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )


def guidepost_2():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        visualize=False,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )


def guidepost_no_store_LDS_1():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        storeLDS_D=False,
        visualize=False,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=0,
        **fp16,
    )


def guidepost_no_store_LDS_2():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        storeLDS_D=False,
        visualize=False,
        scheduler="Priority",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=0,
        **fp16,
    )


def tensile_asm_guidepost_1():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8448,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        visualize=False,
        scheduler="TENSILE_ASM",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )


def tensile_asm_guidepost_2():
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=256,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=2,
        trans_A="N",
        trans_B="T",
        visualize=False,
        scheduler="TENSILE_ASM",
        prefetch=True,
        prefetchInFlight=2,
        prefetchLDSFactor=2,
        **fp16,
    )


def tensile_guidepost():
    yield TensileRun(
        config=str(
            repo_dir
            / "test"
            / "unit"
            / "GemmGuidePost"
            / "HGemmGuidePost_Optimized.yaml"
        ),
    )


def tensile_sgemm_guidepost():
    yield TensileRun(
        config=str(
            repo_dir
            / "test"
            / "unit"
            / "GemmGuidePost"
            / "GemmGuidePost_Optimized.yaml"
        ),
    )


def streamk():
    for twoTile in {True, False}:
        # SGEMM
        yield GEMMRun(
            M=3072,
            N=4096,
            K=4096,
            mac_m=64,
            mac_n=64,
            mac_k=64,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            # prefetchInFlight=2,
            # prefetchLDSFactor=2,
            streamK=True,
            streamKTwoTile=twoTile,
            **fp32,
        )
        # HGEMM
        yield GEMMRun(
            M=7680,
            N=8448,
            K=8448,
            mac_m=64,
            mac_n=64,
            mac_k=64,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            # prefetchInFlight=2,
            # prefetchLDSFactor=2,
            streamK=True,
            streamKTwoTile=twoTile,
            **fp16,
        )
        yield GEMMRun(
            M=7680,
            N=8448,
            K=8192,
            mac_m=64,
            mac_n=64,
            mac_k=64,
            workgroup_size_x=128,
            workgroup_size_y=2,
            trans_A="N",
            trans_B="T",
            visualize=False,
            prefetch=False,  # TODO: Fix k loop unrolling with stream k
            # prefetchInFlight=2,
            # prefetchLDSFactor=2,
            streamK=True,
            streamKTwoTile=twoTile,
            **fp16,
        )
    yield GEMMRun(
        M=7680,
        N=8448,
        K=8192,
        mac_m=128,
        mac_n=128,
        mac_k=16,
        workgroup_size_x=128,
        workgroup_size_y=4,
        trans_A="N",
        trans_B="T",
        visualize=False,
        prefetch=False,  # TODO: Fix k loop unrolling with stream k
        # prefetchInFlight=2,
        # prefetchLDSFactor=2,
        streamK=True,
        streamKTwoTile=False,
        numWGs=220,
        **fp16,
    )


def tensile_benchmarks():
    yield from tensile_guidepost()
    yield from tensile_sgemm_guidepost()
    yield from tensile_asm_guidepost_1()
    yield from tensile_asm_guidepost_2()


def codegen():
    yield CodeGenRun(instCount=40000, instructions="comments")
    yield CodeGenRun(instCount=40000, instructions="simple_mfma")
    yield CodeGenRun(instCount=40000, instructions="complex_mfma_with_coop")


def all():
    yield from sgemm()
    yield from hgemm()
    yield from guidepost_no_store_LDS_1()
    yield from guidepost_no_store_LDS_2()
    yield from codegen()
    yield from tensile_benchmarks()
    yield from streamk()


def hgemm_guideposts():
    yield from guidepost_1()
    yield from guidepost_2()
    yield from tensile_asm_guidepost_1()
    yield from tensile_asm_guidepost_2()


def priority_problems():
    return {
        "1. HGEMM Guidepost": {
            "M": 7680,
            "N": 8448,
            "trans_A": "N",
            "trans_B": "T",
        },
        "2. Halfs": {"type_A": "half"},
        "3. Floats": {"type_A": "float"},
    }
