#!/usr/bin/env groovy
@Library('rocJenkins@pong') _
import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;
import jenkins.*;

def runCI =
{
    nodeDetails, jobName->

    def prj = new rocProject('rocRoller', 'PreCheckin')

    //use docker files from this repo
    prj.repoDockerfile = true
    prj.defaults.ccache = false

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

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        String mxDataGeneratorGitURL = params?.ROCROLLER_MXDATAGENERATOR_GIT_URL ?: baseParams?.ROCROLLER_MXDATAGENERATOR_GIT_URL
        String mxDataGeneratorGitTag = params?.ROCROLLER_MXDATAGENERATOR_GIT_TAG ?: baseParams?.ROCROLLER_MXDATAGENERATOR_GIT_TAG
        def useYAMLCPP = !platform.jenkinsLabel.contains('clang')
        commonGroovy.runCompileCommand(platform, project, jobName, mxDataGeneratorGitURL, mxDataGeneratorGitTag, false, false, '', useYAMLCPP)
    }

    def testCommand =
    {
        platform, project->

        commonGroovy.runTestCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
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
        string(
            name: "ROCROLLER_MXDATAGENERATOR_GIT_URL",
            defaultValue: params?.ROCROLLER_MXDATAGENERATOR_GIT_URL ?: "",
            trim: true,
            description: "Specify the specific mxDataGenerator Git URL"
        ),
        string(
            name: "ROCROLLER_MXDATAGENERATOR_GIT_TAG",
            defaultValue: params?.ROCROLLER_MXDATAGENERATOR_GIT_TAG ?: "",
            trim: true,
            description: "Specify the specific mxDataGenerator tag/commit hash"
        ),
        booleanParam(
            name: "Unique Docker image tag",
            defaultValue: false,
            description: "Whether to tag the built docker image with a unique tag. WARNING: Use sparingly, each unique tag costs significant storage space."
        )
    ]
    auxiliary.registerAdditionalParameters(additionalParameters)

    def propertyList = [
        "enterprise":[pipelineTriggers([cron('0 1 * * 0')])],
        "rocm-libraries":[pipelineTriggers([cron('0 1 * * 0')])]
    ]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = [
        "enterprise":([
            "rocroller-ubuntu20-clang":['rocroller-compile', 'rocroller-gfx90a'],
            "rocroller-ubuntu20-gcc":['rocroller-compile', 'rocroller-gfx90a']
        ]),
        "rocm-libraries":([
            "rocroller-ubuntu20-clang":['rocroller-compile', 'rocroller-gfx90a'],
            "rocroller-ubuntu20-gcc":['rocroller-compile', 'rocroller-gfx90a']
        ])
    ]
    jobNameList = auxiliary.appendJobNameList(jobNameList)

    propertyList.each
    {
        jobName, property->
        if (urlJobName == jobName)
            properties(auxiliary.addCommonProperties(property))
    }

    jobNameList.each
    {
        jobName, nodeDetails->
        if (urlJobName == jobName)
            runCI(nodeDetails, jobName)
    }

    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
        stage(urlJobName) {
            runCI(["rocroller-ubuntu20-clang":['rocroller-compile']], urlJobName)
        }
    }
}
