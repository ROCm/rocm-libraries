"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/hipblas": "blas",
    "projects/hipblas-common": "blas",
    "projects/hipblaslt": "blas",
    "projects/hipcub": "prim",
    "projects/hiprand": "rand",
    "projects/rocblas": "blas",
    "projects/rocprim": "prim",
    "projects/rocrand": "rand",
    "projects/rocthrust": "prim",
    "shared/rocroller": "blas",
    "shared/tensile": "blas"
}

project_map = {
    "prim": {
        "cmake_options": "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocprim, rocthrust, hipcub",
        "subtree_checkout": "projects/rocprim\nprojects/hipcub\nprojects/rocthrust",
    },
    "rand": {
        "cmake_options": "-DTHEROCK_ENABLE_RAND=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocrand, hiprand",
        "subtree_checkout": "projects/rocrand\nprojects/hiprand",
    },
    "blas": {
        "cmake_options": "-DTHEROCK_ENABLE_BLAS=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "hipblaslt, rocblas",
        "subtree_checkout": "projects/hipblaslt\nprojects/hipblas-common\nprojects/rocblas\nprojects/hipblas\nshared/mxdatagenerator\nshared/rocroller\nshared/tensile",
    }
}
