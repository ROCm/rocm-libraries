#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, boolean debug=false)
{
    project.paths.construct_build_prefix()
    
    def command = """#!/usr/bin/env bash
            set -x
            ${project.paths.project_build_prefix}/docs/run_doc.sh
            """

    try
    {
        platform.runCommand(this, command)
    }
    catch(e)
    {
        throw e
    }

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/docs/_build/html",
                reportFiles: "index.html",
                reportName: "Documentation",
                reportTitles: "Documentation"])
}

def runCI = 
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocThrust', 'StaticAnalysis')

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false
    boolean staticAnalysis = true

    def compileCommand =
    {
        platform, project->

        runCompileCommand(platform, project, jobName, false)
    }

    buildProject(prj , formatCheck, nodes.dockerArray, compileCommand, null, null, staticAnalysis)
}

ci: { 
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
    stage(urlJobName) {
        runCI([ubuntu18:['any']], urlJobName)
    }
}
