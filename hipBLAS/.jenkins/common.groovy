// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName, boolean sameOrg=false)
{
    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, sameOrg)
        }
    }
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("noSolver"))
        {
            project.paths.build_command = project.paths.build_command.replaceAll(' -c', ' -cn')
        }

        if (pullRequest.labels.contains("debug"))
        {
            project.paths.build_command = project.paths.build_command.replaceAll(' -c', ' -cg')
        }
    }

    String centos = platform.jenkinsLabel.contains('centos7') ? 'source scl_source enable devtoolset-7' : ':'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${centos}
                LD_LIBRARY_PATH=/opt/rocm/lib ${project.paths.build_command}
                """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String stagingDir = "${project.paths.project_build_prefix}/build/release/clients/staging"

    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("debug"))
        {
            stagingDir = "${project.paths.project_build_prefix}/build/debug/clients/staging"
        }
    }

    String gtestCommonEnv = "HIPBLAS_CLIENT_RAM_GB_LIMIT=95"
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${stagingDir}
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib ${gtestCommonEnv} GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_output=xml --gtest_color=yes
                """

    platform.runCommand(this, command)

    // In an upcoming release, we are replacing hipblasDatatype_t with hipDataType. We have created hipblas_v2-test to test the new
    // interfaces while hipblasDatatype_t is deprecated. Thus, hipblas-test will be testing the old, deprecated, functions
    // using hipblasDatatype_t, and hipblas_v2-test will be testing the upcoming interfaces.
    def v2TestCommand = """#!/usr/bin/env bash
                    set -x
                    cd ${stagingDir}
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib ${gtestCommonEnv} GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas_v2-test --gtest_output=xml --gtest_color=yes
                """

    platform.runCommand(this, v2TestCommand)

    def yamlTestCommand = """#!/usr/bin/env bash
                    set -x
                    cd ${stagingDir}
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib ${gtestCommonEnv} GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./hipblas-test --gtest_output=xml --gtest_color=yes --yaml hipblas_smoke.yaml
                """
    platform.runCommand(this, yamlTestCommand)
    junit "${stagingDir}/*.xml"
}

def runPackageCommand(platform, project, jobName, label='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''
    String ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    String dir = jobName.contains('Debug') ? "debug" : "release"
    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        if (pullRequest.labels.contains("debug"))
        {
            dir = "debug"
        }
    }

    command = """
            set -x
            cd ${project.paths.project_build_prefix}/build/${dir}
            make package
            mkdir -p package
            if [ ! -z "$label" ]
            then
                for f in hipblas*.$ext
                do
                    mv "\$f" "hipblas${label}-\${f#*-}"
                done
            fi
            mv *.${ext} package/
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dir}/package/*.${ext}""")
}

return this
