// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    String compiler = 'amdclang++'
    String hipClangArgs = jobName.contains('hipclang') ? ' --hip-clang' : ''
    String staticArgs = jobName.contains('static') ? ' -s' : ''
    //Temporary workaround due to bug in container
    String centos7Workaround = platform.jenkinsLabel.contains('centos7') ? 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib64/' : ''

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, sameOrg)
        }
    }
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                ${centos7Workaround}
                ${auxiliary.gfxTargetParser()}
                CXX=/opt/rocm/bin/${compiler} ${project.paths.build_command} ${hipClangArgs} ${staticArgs}
            """

    platform.runCommand(this, command)
}


def runTestCommand (platform, project, gfilter, boolean rocmExamples=false, String dirmode = "release")
{
    //Temporary workaround due to bug in container
    String centos7Workaround = platform.jenkinsLabel.contains('centos7') ? 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib64/' : ''

    def hmmTestCommand= ''
    if (platform.jenkinsLabel.contains('gfx90a'))
    {
        hmmTestCommand = """
                            HSA_XNACK=0 GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml:test_detail_hmm_xnack_off.xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                            HSA_XNACK=1 GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml:test_detail_hmm_xnack_on.xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                         """
    }

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${dirmode}/clients/staging
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                ${centos7Workaround}
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./rocsparse-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
                ${hmmTestCommand}
            """

    platform.runCommand(this, command)
    //ROCM-Examples
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
                    ${buildString}
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

def runTestWithSanitizerCommand (platform, project, gfilter, String dirmode = "release")
{
    //Temporary workaround due to bug in container
    String centos7Workaround = platform.jenkinsLabel.contains('centos7') ? 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib64/' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/${dirmode}/clients/staging
		        export ASAN_LIB_PATH=\$(/opt/rocm/llvm/bin/clang -print-file-name=libclang_rt.asan-x86_64.so)
                export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:\$(dirname "\${ASAN_LIB_PATH}")
                ${centos7Workaround}
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG ASAN_SYMBOLIZER_PATH=/opt/rocm/llvm/bin/llvm-symbolizer ASAN_OPTIONS=detect_leaks=1 LSAN_OPTIONS=suppressions=../../../../suppr.txt ./rocsparse-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}-*known_bug*
            """

    platform.runCommand(this, command)
}

def runCoverageCommand (platform, project, gfilter, String dirmode = "release")
{
    //Temporary workaround due to bug in container
    String centos7Workaround = platform.jenkinsLabel.contains('centos7') ? 'export LD_LIBRARY_PATH=\$LD_LIBRARY_PATH:/opt/rocm/lib64/' : ''

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}/build/${dirmode}
                export LD_LIBRARY_PATH=/opt/rocm/lib/
                ${centos7Workaround}
                GTEST_LISTENER=NO_PASS_LINE_IN_LOG make coverage_cleanup coverage GTEST_FILTER=${gfilter}-*known_bug*
            """

    platform.runCommand(this, command)

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/build/${dirmode}/lcoverage",
                reportFiles: "index.html",
                reportName: "Code coverage report",
                reportTitles: "Code coverage report"])
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
            set -x
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
