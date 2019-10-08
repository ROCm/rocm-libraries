#!/usr/bin/env groovy
@Library('rocJenkins') _
import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocprimCI:
{

    def rocprim = new rocProject('rocPRIM')

    def nodes = new dockerNodes(['gfx803 && centos7', 'ubuntu && gfx908', 'gfx900 && centos7', 'gfx906 && centos7', 'sles'], rocprim)

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
        def testCommand = 'ctest --output-on-failure'

        def command = """#!/usr/bin/env bash
                    set -x
                    cd ${project.paths.project_build_prefix}
                    cd ${project.testDirectory}
                    ${sudo} LD_LIBRARY_PATH=/opt/rocm/lib ${testCommand}
                """

        platform.runCommand(this, command)
    }

    def packageCommand =
    {
        platform, project->

        def command

        if(platform.jenkinsLabel.contains('centos'))
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/${project.testDirectory}
                    make package
                    rm -rf package && mkdir -p package
                    mv *.rpm package/
                    rpm -qlp package/*.rpm
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/${project.testDirectory}/package/*.rpm""")
        }
        else if(platform.jenkinsLabel.contains('hip-clang'))
        {
            packageCommand = null
        }
        else
        {
            command = """
                    set -x
                    cd ${project.paths.project_build_prefix}/${project.testDirectory}
                    make package
                    rm -rf package && mkdir -p package
                    mv *.deb package/
                    dpkg -c package/*.deb
                """

            platform.runCommand(this, command)
            platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/${project.testDirectory}/package/*.deb""")
        }
    }

    buildProject(rocprim, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}
