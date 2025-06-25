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

    def uniqueTag = params?."Unique Docker image tag" ? org.apache.commons.lang.RandomStringUtils.random(9, true, true) : ""

    def baseParams = rocRollerGetBaseParameters()

    def nodes = new dockerNodes(nodeDetails, jobName, prj)
    nodes.dockerArray.each {
        _, docker ->
        // parameters inherited from target job
        ["ROCROLLER_AMDGPU_URL", "ROCROLLER_AMDGPU_BUILD_NUMBER", "ROCROLLER_AMDGPU_BUILD_URI"].each {
            param ->
            def value = params?."${param}" ?: baseParams?."${param}";
            if (value)
            {
                docker.buildArgs += " --build-arg ${param}=${value}"
            }
        }

        if (uniqueTag)
        {
            docker.customFinalTag = uniqueTag
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

def rocRollerGetBaseParameters() {
    def baseParameters = jenkins.model.Jenkins.instance.getItemByFullName(env.JOB_NAME)
        .parent.getJob(env.CHANGE_TARGET)
        ?.getProperty(hudson.model.ParametersDefinitionProperty)
        ?.parameterDefinitions
        ?.collect {[ it.name, it.defaultParameterValue.value]}
        ?.collectEntries();
    return baseParameters;
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def additionalParameters = [
        string(
            name: "ROCROLLER_AMDGPU_URL",
            defaultValue: params?.ROCROLLER_AMDGPU_URL ?: "",
            trim: true,
            description: "URL to retrieve AMDGPU install package from"
        ),
        string(
            name: "ROCROLLER_AMDGPU_BUILD_NUMBER",
            defaultValue: params?.ROCROLLER_AMDGPU_BUILD_NUMBER ?: "",
            trim: true,
            description: "Build number to use for AMDGPU"
        ),
        string(
            name: "ROCROLLER_AMDGPU_BUILD_URI",
            defaultValue: params?.ROCROLLER_AMDGPU_BUILD_URI ?: "",
            trim: true,
            description: "Specify the specific artifact path for AMDGPU"
        ),
        booleanParam(
            name: "Unique Docker image tag",
            defaultValue: false,
            description: "Whether to tag the built docker image with a unique tag. WARNING: Use sparingly, each unique tag costs significant storage space."
        )
    ]
    auxiliary.registerAdditionalParameters(additionalParameters)

    properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 12 * * 6')])]))

    def jobNameList = ["enterprise":(["ubuntu20":['rocroller-compile']])]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }
}
