// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    //Set CI node's gfx arch as target if PR, otherwise use default targets of the library
    String amdgpuTargets = env.BRANCH_NAME.startsWith('PR-') ? '-DAMDGPU_TARGETS=\$gfx_arch' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${auxiliary.gfxTargetParser()}
                ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/hipcc ${buildTypeArg} ${amdgpuTargets} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ../..
                make -j\$(nproc)
                """

    platform.runCommand(this, command)
}


def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def testCommand = "ctest --output-on-failure "
    def testCommandExcludeRegex = /(rocprim.device_reduce_by_key|rocprim.device_radix_sort)/
    def testCommandExclude = "--exclude-regex \"${testCommandExcludeRegex}\""
    def hmmExcludeRegex = /(rocprim.device_scan|rocprim.device_reduce_by_key|rocprim.block_sort_bitonic|rocprim.device_merge|rocprim.device_merge_sort|rocprim.device_partition|rocprim.device_segmented_radix_sort|rocprim.device_segmented_scan)/
    def hmmTestCommandExclude = "--exclude-regex \"${hmmExcludeRegex}\""
    def hmmTestCommand = ''
    if (platform.jenkinsLabel.contains('gfx90a'))
    {
        echo("HMM TESTS DISABLED")
        /*hmmTestCommand = """
                            export HSA_XNACK=1
                            export ROCPRIM_USE_HMM=1
                            ${testCommand} ${hmmTestCommandExclude}
                         """*/
    }
    echo(env.JOB_NAME)
    if (env.JOB_NAME.contains('bleeding-edge'))
    {
        testCommand = ''
        testCommandExclude = ''
        hmmTestCommand = ''
        echo("TESTS DISABLED")
    }
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                cd ${project.testDirectory}
                ${testCommand} ${testCommandExclude}
                if (( \$? != 0 )); then
                    exit 1
                fi
                ${hmmTestCommand}
            """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this

