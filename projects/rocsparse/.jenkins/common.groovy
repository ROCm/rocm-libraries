// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    String compiler = 'amdclang++'
    String hipClangArgs = jobName.contains('hipclang') ? ' --hip-clang' : ''
    String staticArgs = jobName.contains('static') ? ' -s' : ''

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, sameOrg)
        }
    }
    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${pullMatricesCommand()}
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                ${auxiliary.gfxTargetParser()}
                CXX=/opt/rocm/bin/${compiler} ${project.paths.build_command} ${hipClangArgs} ${staticArgs}
            """

    platform.runCommand(this, command)
}

// Fetches the DVC-tracked SuiteSparse archives in clients/matrices, which the client tests require
def pullMatricesCommand()
{
    // The pull is skipped when the node's matrices cache already has every converted matrix.
    // The sparse checkout may omit the .dvc directory at the repository root.
    // Jenkins runs the container as a UID without a passwd entry, so dvc needs LOGNAME
    // (https://docs.python.org/3/library/getpass.html#getpass.getuser).
    // No "|| exit 1" after the subshell: it would disable set -e inside it, and the caller's set -e
    // already stops on a failed subshell.
    return """
                (
                    set -e
                    repo_root=\$(git rev-parse --show-toplevel)
                    matrices_cached=true
                    for pointer in "\${repo_root}"/projects/rocsparse/clients/matrices/*.tar.gz.dvc; do
                        if [ ! -f "\${JENKINS_HOME_DIR}/rocsparse_matrices/\$(basename "\${pointer}" .tar.gz.dvc).csr" ]; then
                            matrices_cached=false
                        fi
                    done
                    if [ "\${matrices_cached}" = true ]; then
                        echo "All test matrices are in \${JENKINS_HOME_DIR}/rocsparse_matrices, skipping dvc pull"
                        exit 0
                    fi
                    if [ ! -f "\${repo_root}/.dvc/config" ]; then
                        mkdir -p "\${repo_root}/.dvc"
                        git -C "\${repo_root}" show HEAD:.dvc/config > "\${repo_root}/.dvc/config.tmp"
                        mv "\${repo_root}/.dvc/config.tmp" "\${repo_root}/.dvc/config"
                    fi
                    dvc_cmd=dvc
                    if ! command -v dvc > /dev/null; then
                        dvc_venv="\${PWD}/build/dvc-venv"
                        python3 -m venv "\${dvc_venv}"
                        "\${dvc_venv}/bin/python" -m pip install --quiet 'dvc[s3]==3.67.1'
                        dvc_cmd="\${dvc_venv}/bin/dvc"
                    fi
                    cd "\${repo_root}"
                    LOGNAME=jenkins DVC_NO_ANALYTICS=1 \${dvc_cmd} pull -R projects/rocsparse/clients/matrices
                )
            """
}


def runTestCommand (platform, project, gfilter, boolean rocmExamples=false, String dirmode = "release")
{
    def hmmTestCommand= """GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*"""
    if (platform.jenkinsLabel.contains('gfx90a') || platform.jenkinsLabel.contains('gfx942'))
    {
        hmmTestCommand = """
                            HSA_XNACK=0 GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml:test_detail_hmm_xnack_off.xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                            HSA_XNACK=1 GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml:test_detail_hmm_xnack_on.xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                         """
    }

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${dirmode}
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                # Same shell step as rocsparse-test. Groovy calls before this
                # script are not executed by Math CI.
                cmake --build . --target rocsparse-unit-test rocsparse-unit-test-device
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./clients/staging/rocsparse-unit-test --gtest_output=xml:test_detail_unit.xml --gtest_color=yes
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./clients/staging/rocsparse-unit-test-device --gtest_output=xml:test_detail_unit_device.xml --gtest_color=yes
                cd clients/staging
                ${hmmTestCommand}
            """

    platform.runCommand(this, command)
    //ROCM-Examples
    if (rocmExamples){
        testCommand = """#!/usr/bin/env bash
                    set -ex
                    cd ${project.paths.project_build_prefix}/build/release/package
                    ${auxiliary.installPackagesCommand(platform.jenkinsLabel, false, "")}
                    cd ../../..
                    testDirs=("Libraries/rocSPARSE")
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


// Fast, CPU-only unit tests (rocsparse-unit-test target). No GPU required, so this
// runs early as a quick gate before the GPU-dependent gtest / coverage suites.
// The target is built as part of the standard clients-tests build (install.sh).
def runUnitTestCommand (platform, project, String dirmode = "release")
{
    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${dirmode}
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                cmake --build . --target rocsparse-unit-test
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./clients/staging/rocsparse-unit-test --gtest_output=xml:test_detail_unit.xml --gtest_color=yes
            """

    platform.runCommand(this, command)
}


// GPU unit tests (rocsparse-unit-test-device). The precheckin test stage already
// has a GPU, so this runs there with the host unit tests. install.sh builds the
// clients-tests target; cmake --build ensures this binary exists before launch.
def runUnitTestDeviceCommand (platform, project, String dirmode = "release")
{
    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${dirmode}
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                cmake --build . --target rocsparse-unit-test-device
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./clients/staging/rocsparse-unit-test-device --gtest_output=xml:test_detail_unit_device.xml --gtest_color=yes
            """

    platform.runCommand(this, command)
}


def runHipDebugTestCommand (platform, project, gfilter, String dirmode = "release")
{
    def hmmTestCommand= """GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --test-hip-debug --test-hip-debug-full  --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*"""

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${dirmode}/clients/staging
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                ${hmmTestCommand}
            """

    platform.runCommand(this, command)
}


def runTestWithSanitizerCommand (platform, project, gfilter, String dirmode = "release")
{
    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${dirmode}/clients/staging
		        export ASAN_LIB_PATH=\$(/opt/rocm/llvm/bin/clang -print-file-name=libclang_rt.asan-x86_64.so)
                export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(dirname "\${ASAN_LIB_PATH}")
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG ASAN_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS=suppressions=../../../../suppr.txt ./rocsparse-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
            """

    platform.runCommand(this, command)
}

def runCoverageCommand (platform, project, gfilter, String dirmode = "release")
{
    String commitSha
    String repoUrl
    (commitSha, repoUrl) = util.getGitHubCommitInformation(project.paths.project_src_prefix)

    withCredentials([string(credentialsId: "mathlibs-codecov-token-rocm-libraries", variable: 'CODECOV_TOKEN')])
    {
        def command = """#!/usr/bin/env bash
                    set -ex
                    cd ${project.paths.project_build_prefix}/build/${dirmode}
                    export LD_LIBRARY_PATH=/opt/rocm/lib/
                    # Same shell step as coverage. Groovy calls before this
                    # script are not executed, so a unit-test failure must be
                    # raised here, before make coverage ignores it.
                    cmake --build . --target rocsparse-unit-test rocsparse-unit-test-device
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./clients/staging/rocsparse-unit-test --gtest_output=xml:test_detail_unit.xml --gtest_color=yes
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./clients/staging/rocsparse-unit-test-device --gtest_output=xml:test_detail_unit_device.xml --gtest_color=yes
                    GTEST_LISTENER=NO_PASS_LINE_IN_LOG make coverage_cleanup coverage GTEST_FILTER=${gfilter}-*known_bug*
                    /usr/local/bin/codecov -v -U \$http_proxy -t ${CODECOV_TOKEN} --file coverage-report/coverage.info --name rocm-libraries --flags rocSPARSE --sha ${commitSha}
                """

        platform.runCommand(this, command)
    }

}

def runPackageCommand(platform, project, String dirmode = "release")
{
    def command

    String pkgType
    String pkgInfoCommand
    if(platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles') || platform.jenkinsLabel.contains('rhel') || platform.jenkinsLabel.contains('cs9'))
    {
        pkgType = "rpm"
        pkgInfoCommand = "rpm -qlp package/*.rpm"
    }
    else
    {
        pkgType = "deb"
        pkgInfoCommand = "for pkg in package/*.deb; do dpkg -I \$pkg; dpkg -c \$pkg; done"
    }
    command = """
            set -ex
            cd ${project.paths.project_build_prefix}/build/${dirmode}
            make package
            mkdir -p package
            mv *.${pkgType} package/
            ${pkgInfoCommand}
        """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dirmode}/package/*.${pkgType}""")
}

return this
