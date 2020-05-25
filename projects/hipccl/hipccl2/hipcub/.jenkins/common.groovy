// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

def runCompileCommand(platform, project, jobName)
{
    project.paths.construct_build_prefix()
        
    def command 

    def getRocPRIM = auxiliary.getLibrary('rocPRIM', platform.jenkinsLabel)

    if(jobName.contains('hipclang'))
    {
        command = """#!/usr/bin/env bash
                set -x
                ${getRocPRIM}
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                """
    }
    else
    {
        command = """#!/usr/bin/env bash
                set -x
                ${getRocPRIM}
                cd ${project.paths.project_build_prefix}
                LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hcc ${project.paths.build_command}
                """
    }

    platform.runCommand(this, command)
}

def runTestCommand (platform, project)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String centos = platform.jenkinsLabel.contains('centos') ? '3' : ''
    def testCommand = "ctest${centos} --output-on-failure"

    def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib/ ${testCommand}
                """

    platform.runCommand(this, command)
}

def runPackageCommand(platform, project)
{
    def command
        
    if(platform.jenkinsLabel.contains('ubuntu'))
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                rm -rf package && mkdir -p package
                mv *.deb package/
                dpkg -c package/*.deb
                """        
        
        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.deb""")
    }
    else
    {
        command = """
                set -x
                cd ${project.paths.project_build_prefix}/build/release
                make package
                rm -rf package && mkdir -p package
                mv *.rpm package/
                rpm -qlp package/*.rpm
                """
        
        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/release/package/*.rpm""")
    }
}

return this

