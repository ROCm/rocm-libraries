def overflowed_resources_branch(overflowed_resources: int) -> bool:
    """Pure model of KernelWriterAssembly.py:1902 branch predicate.

    `if self.states.overflowedResources:` -- truthiness of an int field.
    overflowedResources is a StateValues int (KernelWriter.py:183), init 0,
    set to a resource-overflow code in {1..8} by any of 8 upstream write-paths.
    The branch is taken iff the field holds a nonzero code.

    pre: 0 <= overflowed_resources <= 8
    post: __return__ == (overflowed_resources != 0)
    """
    return bool(overflowed_resources)
