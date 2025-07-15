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

    String debug = project.buildName.contains('Debug') ? '-g' : ''
    String centos = platform.jenkinsLabel.contains('centos') ? 'source scl_source enable devtoolset-7' : ':'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                ${centos}
                LD_LIBRARY_PATH=/opt/rocm/lib ${project.paths.build_command} ${debug}
                """
    platform.runCommand(this, command)
}

def runTestCommand(platform, project)
{
    String buildType = project.buildName.contains('Debug') ? "debug" : "release"
    String testExe = "hipsolver-test"
    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/${buildType}/clients/staging
                    LD_LIBRARY_PATH=/opt/rocm/lib GTEST_LISTENER=NO_PASS_LINE_IN_LOG ./${testExe} --gtest_output=xml --gtest_color=yes
                """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project, jobName, label='')
{
    def command

    label = label != '' ? '-' + label.toLowerCase() : ''
    String ext = platform.jenkinsLabel.contains('ubuntu') ? "deb" : "rpm"
    String dir = project.buildName.contains('Debug') ? "debug" : "release"

    String testPackageCommand;
    if (platform.jenkinsLabel.contains('ubuntu'))
    {
        testPackageCommand = 'sudo apt-get install -y --simulate '
    }
    else if (platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('rhel'))
    {
        testPackageCommand = 'sudo yum install -y --setopt tsflags=test '
    }
    else
    {
        testPackageCommand = 'sudo zypper install -y --dry-run --download-only --allow-unsigned-rpm '
    }

    command = """
            set -ex
            cd ${project.paths.project_build_prefix}/build/${dir}
            make package
            ${testPackageCommand} ./hipsolver*.$ext
            mkdir -p package
            if [ ! -z "$label" ]
            then
                for f in hipsolver*.$ext
                do
                    mv "\$f" "hipsolver${label}-\${f#*-}"
                done
            fi
            mv *.${ext} package/
        """

    platform.runCommand(this, command)
    platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/${dir}/package/*.${ext}""")
}

return this
