# Copyright © Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier:  MIT

import numpy as np
import hipdnn_frontend as hipdnn


def run_matmul():
    """
    Minimal sample: build Matmul graph C = A @ B
    """
    print("=" * 70)
    print("Matmul Test")
    print("=" * 70)

    # Dimensions: A [M, K], B [K, N] -> C [M, N]
    M, K, N = 4, 3, 5

    # Create a handle and graph
    print("\nCreating hipdnn handle...")
    handle = hipdnn.create_handle()

    graph = hipdnn.Graph()
    graph.set_name("matmul_graph")
    graph.set_io_data_type(hipdnn.DataType.FLOAT)
    graph.set_intermediate_data_type(hipdnn.DataType.FLOAT)
    graph.set_compute_data_type(hipdnn.DataType.FLOAT)
    print("Graph created with FLOAT data type")

    # Create input tensors
    print("\nCreating tensors...")
    a = hipdnn.Tensor.create([M, K], hipdnn.DataType.FLOAT)
    a.set_name("A")
    b = hipdnn.Tensor.create([K, N], hipdnn.DataType.FLOAT)
    b.set_name("B")
    print(f"  A: shape={[M, K]}, uid={a.get_uid()}")
    print(f"  B: shape={[K, N]}, uid={b.get_uid()}")

    # Matmul attributes
    attrs = hipdnn.MatmulAttributes()
    attrs.set_name("matmul_node")

    print("\nBuilding matmul operation...")
    c = graph.matmul(a, b, attrs)
    if c:
        c.set_name("C")
        c.set_output(True)
        print(f"  C: shape={[M, N]}, uid={c.get_uid()}")

    # Validate, build, and prepare execution
    print("\nValidating graph...")
    validation_result = graph.validate()
    if validation_result.is_good():
        print("✓ Graph validation successful!")
    else:
        print(f"✗ Graph validation failed: {validation_result.get_message()}")
        return

    print("\nBuilding operation graph...")
    build_result = graph.build_operation_graph(handle)
    if build_result.is_good():
        print("✓ Operation graph built successfully")
    else:
        print(f"✗ Failed to build operation graph: {build_result.get_message()}")
        return


if __name__ == "__main__":
    try:
        run_matmul()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
