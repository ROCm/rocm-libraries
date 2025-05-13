// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean staticLibrary=false)
{
    project.paths.construct_build_prefix()

    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug' : '-DCMAKE_BUILD_TYPE=Release'
    String buildStatic = staticLibrary ? '-DBUILD_SHARED_LIBS=OFF' : '-DBUILD_SHARED_LIBS=ON'
    String buildTypeDir = debug ? 'debug' : 'release'
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'
    //Set CI node's gfx arch as target if PR, otherwise use default targets of the library
    String amdgpuTargets = env.BRANCH_NAME.startsWith('PR-') ? '-DAMDGPU_TARGETS=\$gfx_arch' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                mkdir -p build/${buildTypeDir} && cd build/${buildTypeDir}
                ${auxiliary.gfxTargetParser()}
                ${cmake} --toolchain=toolchain-linux.cmake ${buildTypeArg} ${buildStatic} ${amdgpuTargets} -DBUILD_TEST=ON -DBUILD_BENCHMARK=ON ../..
                make -j\$(nproc)
                """

    platform.runCommand(this, command)
}


def runTestCommand (platform, project, boolean rocmExamples=false)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)

    def testCommand = "ctest --output-on-failure "
    def testCommandExcludeRegex = /(rocprim.block_histogram)/
    def testCommandExclude = "--exclude-regex \"${testCommandExcludeRegex}\""
    def hmmExcludeRegex = ''
    def hmmTestCommandExclude = "--exclude-regex \"${hmmExcludeRegex}\""
    def hmmTestCommand = ''
    if (platform.jenkinsLabel.contains('gfx90a'))
    {
        hmmTestCommand = """
                            export HSA_XNACK=1
                            export ROCPRIM_USE_HMM=1
                            ${testCommand} ${hmmTestCommandExclude}
                         """
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
    //ROCM Examples
    if (rocmExamples){
        String buildString = ""
        if (platform.os.contains("ubuntu")){
            buildString += "sudo dpkg -i *.deb"
        }
        else {
            buildString += "sudo rpm -i *.rpm"
        }
        testCommand = """#!/usr/bin/env bash
                    set -ex
                    cd ${project.paths.project_build_prefix}/build/release/package
                    ls
                    ${buildString}
                    cd ../../..
                    testDirs=("Libraries/rocPRIM")
                    git clone https://github.com/ROCm/rocm-examples.git
                    rocm_examples_dir=\$(readlink -f rocm-examples)
                    for testDir in \${testDirs[@]}; do
                        cd \${rocm_examples_dir}/\${testDir}
                        cmake -S . -B build
                        cmake --build build
                        cd ./build
                        ctest --output-on-failure
                    done
                """
        platform.runCommand(this, testCommand, "ROCM Examples")  

    }
}

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")

    platform.runCommand(this, packageHelper[0])
        platform.archiveArtifacts(this, packageHelper[1])
}

return this
