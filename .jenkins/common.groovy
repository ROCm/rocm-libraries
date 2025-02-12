// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false, boolean isStatic=false)
{
    project.paths.construct_build_prefix()

    String compiler = 'hipcc'
    String hipClang = ''
    String debug = project.buildName.contains('Debug') ? '-g' : ''
    String centos = platform.jenkinsLabel.contains('centos') ? 'source scl_source enable devtoolset-7' : ''
    List<String> options = []
    Boolean withSparse = true

    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        pullRequest.labels.each
        {
            if (it == "noOptimizations")
            {
                options << "-n"
            }
            else if (it == "ci:no-sparse")
            {
                options << "--no-sparse"
                withSparse = false
            }
        }
    }

    List<String> getDeps = []
    getDeps << auxiliary.getLibrary('hipBLAS-common', platform.jenkinsLabel, null, sameOrg)
    if (isStatic == false)
    {
        getDeps << auxiliary.getLibrary('hipBLASLt', platform.jenkinsLabel, null, sameOrg)
    }
    getDeps << auxiliary.getLibrary('rocBLAS', platform.jenkinsLabel, null, sameOrg)
    if (withSparse)
    {
        getDeps << auxiliary.getLibrary('rocSPARSE', platform.jenkinsLabel, null, sameOrg)
    }
    getDeps << auxiliary.getLibrary('rocPRIM', platform.jenkinsLabel, null, sameOrg)
    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDeps.join('\\n')}
                ${auxiliary.exitIfNotSuccess()}
                ${centos}
                ${project.paths.build_command} ${hipClang} ${debug} ${options.join(' ')}
                ${auxiliary.exitIfNotSuccess()}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, gfilter)
{
    String buildType = project.buildName.contains('Debug') ? 'debug' : 'release'
    String hmmTestCommand = platform.jenkinsLabel.contains('gfx90a') ? 'HSA_XNACK=1 ./rocsolver-test --gtest_filter=*MANAGED_MALLOC* || true' : ''

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}/build/${buildType}/clients/staging
                ./rocsolver-test --gtest_output=xml --gtest_color=yes --gtest_filter=${gfilter}
                if [ -f ./test-rocsolver-dlopen ]; then
                  ./test-rocsolver-dlopen --gtest_color=yes
                fi
                ${hmmTestCommand}
                cd ../..
                CTEST_OUTPUT_ON_FAILURE=1 ctest -R '^test-rocsolver-bench'
                """


    platform.runCommand(this, command)
    junit "${project.paths.project_build_prefix}/build/${buildType}/clients/staging/*.xml"
}

def runPackageCommand(platform, project)
{
    String buildType = project.buildName.contains('Debug') ? 'debug' : 'release'
    def packageHelper = platform.makePackage(platform.jenkinsLabel, "${project.paths.project_build_prefix}/build/${buildType}")
    platform.runCommand(this, packageHelper[0])
    platform.archiveArtifacts(this, packageHelper[1])
}

return this
