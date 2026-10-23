# Copyright 2021 Advanced Micro Devices, Inc.
# This conanfile is used to install development requirements,
# e.g.
#   conan install -o clients=True -if build/deps .

from conans import ConanFile, CMake

class ConanPkgReqs(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake_find_package"
    options = {
        "shared": [True, False],
        "clients": [True, False],
    }
    default_options = {
        "shared": True,
        "clients": False,
    }

    def requirements(self):
        if self.options.clients:
            self.requires("gtest/1.11.0")
            self.requires("benchmark/1.5.2")
