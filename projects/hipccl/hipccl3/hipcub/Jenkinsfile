#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

hipCUBCI:
{

    def hipcub = new rocProject('hipCUB')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['ubuntu && gfx803', 'sles && gfx906', 'gfx900 && centos7', 'gfx906 && centos7'], hipcub)

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        
        def command 

        if(platform.jenkinsLabel.contains('hip-clang'))
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hipcc ${project.paths.build_command} --hip-clang
                    """
        }
        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hcc ${project.paths.build_command}
                    """
        }

        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        String sudo = auxiliary.sudo(platform.jenkinsLabel)

	def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib/ ctest --output-on-failure
                """

        platform.runCommand(this, command)
    }

    def packageCommand =
    {
        platform, project->

        def command
        
        if(platform.jenkinsLabel.contains('hip-clang'))
        {
            packageCommand = null
        }
        else if(platform.jenkinsLabel.contains('ubuntu'))
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

    buildProject(hipcub, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)
}

