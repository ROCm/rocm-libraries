"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
monorepo_map = {
    "projects/rocprim": {
        "cmake_options": "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "test_rocprim",
        "subtree_checkout": "projects/rocprim\nprojects/hipcub\nprojects/rocthrust",
    }
}
