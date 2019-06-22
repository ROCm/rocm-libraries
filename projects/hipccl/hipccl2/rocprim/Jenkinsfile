#!/usr/bin/env groovy
@Library('rocJenkins') _
import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;

rocprimCI:
{

    def rocprim = new rocProject('rocprim')

    def nodes = new dockerNodes(['gfx803', 'gfx900', 'gfx906'], rocprim)

    boolean formatCheck = false
     
    def compileCommand =
    {
        platform, project->

        project.paths.construct_build_prefix()
        def command = """#!/usr/bin/env bash
                  set -x
                  cd ${project.paths.project_build_prefix}
                  LD_LIBRARY_PATH=/opt/rocm/hcc/lib CXX=${project.compiler.compiler_path} ${project.paths.build_command}
                """
        platform.runCommand(this, command)
    }

    def testCommand =
    {
        platform, project->

        def testCommand = 'ctest --output-on-failure -E rocprim.hip.device_merge_sort'

        def command = """#!/usr/bin/env bash
                        set -x
                        cd ${project.paths.project_build_prefix}
                        cd ${project.testDirectory}
                        LD_LIBRARY_PATH=/opt/rocm/hcc/lib ${testCommand}
                  """

        platform.runCommand(this, command)
    }

    def packageCommand =
    {
        platform, project->

        def command = """
                      set -x
                      cd ${project.paths.project_build_prefix}/build
                      make package
                      rm -rf package && mkdir -p package
                      mv *.deb package/
                      dpkg -c package/*.deb
                      """

        platform.runCommand(this, command)
        platform.archiveArtifacts(this, """${project.paths.project_build_prefix}/build/package/*.deb""")
    }

    buildProject(rocprim, formatCheck, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}


