// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    def command
    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, 'develop', sameOrg)
        }
    }
    String centos7 = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'
    
    command = """#!/usr/bin/env bash
                set -x
                ${centos7}
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                CXX=${project.compiler.compiler_path} ${project.paths.build_command}
            """
 
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release/clients/staging
                    ${sudo} GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipsparse-test --gtest_also_run_disabled_tests --gtest_output=xml --gtest_color=yes #--gtest_filter=${gfilter}-*known_bug*
                """

    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/release/clients/staging/*.xml"
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

def runPackageCommand(platform, project)
{
    def packageHelper = platform.makePackage(platform.jenkinsLabel,"${project.paths.project_build_prefix}/build/release")
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this

