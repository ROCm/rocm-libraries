"""
This dictionary is used to map specific file directory changes to the corresponding build flag and tests
"""
subtree_to_project_map = {
    "projects/rocprim": "prim",
    "projects/rocthrust": "prim",
    "projects/hipcub": "prim"
}

project_map = {
    "prim": {
        "cmake_options": "-DTHEROCK_ENABLE_PRIM=ON -DTHEROCK_ENABLE_ALL=OFF",
        "project_to_test": "rocprim, rocthrust, hipcub",
        "subtree_checkout": "projects/rocprim\nprojects/hipcub\nprojects/rocthrust",
    },
}
