#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins') _

// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

rocThrustCI:
{

    def rocthrust = new rocProject('rocThrust')
    // customize for project
    rocthrust.paths.build_command = './install'

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(['ubuntu', 'sles', 'gfx900 && centos7', 'gfx906 && centos7'], rocthrust)

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
                    LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=/opt/rocm/bin/hcc ${project.paths.build_command} -c
                """
 
        }
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def command
        
        if(platform.jenkinsLabel.contains('centos') || platform.jenkinsLabel.contains('sles'))
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    sudo ctest --output-on-failure
                """
        }
        else
        {
            command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}/build/release
                    make -j4
                    ctest --output-on-failure
                """
        }

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

    buildProject(rocthrust, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
