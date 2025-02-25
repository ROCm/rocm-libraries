#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path


def runCI =
{
    nodeDetails, jobName ->

    def prj = new rocProject('rocRoller', 'StaticAnalysis')

    def nodes = new dockerNodes(nodeDetails, jobName, prj)
    nodes.dockerArray.each {
        _, docker ->
        if (params?.AMDGPU_URL)
        {
            docker.buildArgs += " --build-arg AMDGPU_URL=${params.AMDGPU_URL}"
        }
        if (params?.AMDGPU_BUILD_NUMBER)
        {
            docker.buildArgs += " --build-arg AMDGPU_BUILD_NUMBER=${params.AMDGPU_BUILD_NUMBER}"
        }
        if (params?.AMDGPU_BUILD_URI)
        {
            docker.buildArgs += " --build-arg AMDGPU_BUILD_URI=${params.AMDGPU_BUILD_URI}"
        }
    }

    boolean formatCheck = true
    boolean staticAnalysis = false

    def commonGroovy

    def compileCommand =
    {
        platform, project->

        runCompileCommand(platform, project, jobName, false)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, null, null, null, staticAnalysis)

}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def additionalParameters = [
        string(name: "AMDGPU_URL", defaultValue: params?.AMDGPU_URL ?: "", description: "URL to retrieve AMDGPU install package from"),
        string(name: "AMDGPU_BUILD_NUMBER", defaultValue: params?.AMDGPU_BUILD_NUMBER ?: "", description: "Build number to use for AMDGPU"),
        string(name: "AMDGPU_BUILD_URI", defaultValue: params?.AMDGPU_BUILD_URI ?: "", description: "Specify the specific artifact path for AMDGPU")
    ]
    auxiliary.registerAdditionalParameters(additionalParameters)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 12 * * 6')])]))

    def jobNameList = ["enterprise":(["ubuntu20":['rocroller-compile']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }
}
