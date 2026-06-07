def shiftptr_edge_taken(
    tdm_inst: int,
    buffer_load: bool,
    af0em: int,
    grvwa: int,
    af1em: int,
    grvwb: int,
    bias_src: str,
    tlua: bool,
    tlub: bool,
) -> bool:
    # Pure model of KernelWriterAssembly.py:16798 branch predicate.
    # Returns True iff (kernel["EdgeType"] == "ShiftPtr") and enableEdge.
    # Mirrors Solution.py:4992-4995 (EdgeType), 4570-4578 (GuaranteeNoPartial),
    # KernelWriterAssembly.py:16790-16797 (enableEdge).
    enable_tdma = bool(tdm_inst & 1)
    enable_tdmb = bool(tdm_inst & 2)
    edge_is_shiftptr = not (enable_tdma and enable_tdmb)
    gnp_a = (af0em % grvwa == 0) if tlua else True
    gnp_b = (af1em % grvwb == 0) if tlub else True
    enable_edge = False
    if not (buffer_load and gnp_a) and bias_src == "A":
        enable_edge = True
    if not (buffer_load and gnp_b) and bias_src == "B":
        enable_edge = True
    return edge_is_shiftptr and enable_edge


def _ch_total(
    tdm_inst: int,
    buffer_load: bool,
    af0em: int,
    grvwa: int,
    af1em: int,
    grvwb: int,
    bias_src: str,
    tlua: bool,
    tlub: bool,
) -> bool:
    """CrossHair target: function is total over the seeded domain (returns a bool, never raises).

    pre: tdm_inst == 0 or tdm_inst == 3
    pre: af0em == 1 or af0em == 4
    pre: grvwa == 1 or grvwa == 4
    pre: af1em == 1 or af1em == 4
    pre: grvwb == 1 or grvwb == 4
    pre: bias_src == "A" or bias_src == "B" or bias_src == "D"
    post: __return__ is True or __return__ is False
    """
    return shiftptr_edge_taken(
        tdm_inst, buffer_load, af0em, grvwa, af1em, grvwb, bias_src, tlua, tlub
    )
