#!/usr/bin/env groovy
@Library('rocJENKINS.develop') _
import com.amd.project.*
import com.amd.docker.*

////////////////////////////////////////////////////////////////////////
// Mostly generated from snippet generator 'properties; set job properties'
// Time-based triggers added to execute nightly tests, eg '30 2 * * *' means 2:30 AM
properties([
    pipelineTriggers([cron('0 1 * * *'), [$class: 'PeriodicFolderTrigger', interval: '5m']]),
    buildDiscarder(logRotator(
      artifactDaysToKeepStr: '',
      artifactNumToKeepStr: '',
      daysToKeepStr: '',
      numToKeepStr: '10')),
    disableConcurrentBuilds(),
    // parameters([booleanParam( name: 'push_image_to_docker_hub', defaultValue: false, description: 'Push rocprim image to rocl docker-hub' )]),
    [$class: 'CopyArtifactPermissionProperty', projectNames: '*']
   ])


////////////////////////////////////////////////////////////////////////
import java.nio.file.Path;



rocprimCI:
{

    def rocprim = new rocProject('rocprim')

    def nodes = new dockerNodes(['gfx900', 'gfx906'], rocprim.paths)

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
                      echo "rocPRIM is a header only library and does not need packaging"
                      """

        platform.runCommand(this, command)
    }

    buildProject(rocprim, nodes.dockerArray, compileCommand, testCommand, packageCommand)

}


