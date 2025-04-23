// Copyright (c) 2020-2023 Advanced Micro Devices, Inc. All rights reserved.
// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${project.paths.build_command}
            """

    platform.runCommand(this,command)
}

def runTestCommand (platform, project, gfilter, envars)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def command = """#!/usr/bin/env bash
                set -x
                export RUN_TEST_ROOT=\$(pwd)
                cd ${project.paths.project_build_prefix}/build/release/test
                ${sudo} ulimit -l unlimited
                ulimit -a
                ${sudo} ${envars} LD_LIBRARY_PATH=\${RUN_TEST_ROOT}/${project.paths.project_build_prefix}/build/release:\${LD_LIBRARY_PATH} RCCL_ENABLE_SIGNALHANDLER=1 NCCL_DEBUG=INFO HSA_FORCE_FINE_GRAIN_PCIE=1 UT_MULTITHREAD=1 UT_PROCESS_MASK=1 ./rccl-UnitTests --gtest_filter=${gfilter} --gtest_output=xml --gtest_color=yes
            """

   platform.runCommand(this, command)
   junit "${project.paths.project_build_prefix}/build/release/test/*.xml"
}

def runPackageCommand(platform, project, jobName)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
