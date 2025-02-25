#!/usr/bin/env groovy
@Library('rocJenkins@pong') _
import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path;

def runCI =
{
    nodeDetails, jobName->

    def prj = new rocProject('rocRoller', 'PreCheckin')

    //use docker files from this repo
    prj.repoDockerfile = true
    prj.defaults.ccache = true

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

    def commonGroovy

    boolean formatCheck = false

    def compileCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        def useYAMLCPP = !platform.jenkinsLabel.contains('clang')
        commonGroovy.runCompileCommand(platform, project, jobName, false, false, '', useYAMLCPP)
    }

    def testCommand =
    {
        platform, project->

        commonGroovy.runTestCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def additionalParameters = [
        string(name: "AMDGPU_URL", defaultValue: params?.AMDGPU_URL ?: "", description: "URL to retrieve AMDGPU install package from"),
        string(name: "AMDGPU_BUILD_NUMBER", defaultValue: params?.AMDGPU_BUILD_NUMBER ?: "", description: "Build number to use for AMDGPU"),
        string(name: "AMDGPU_BUILD_URI", defaultValue: params?.AMDGPU_BUILD_URI ?: "", description: "Specify the specific artifact path for AMDGPU")
    ]
    auxiliary.registerAdditionalParameters(additionalParameters)

    def propertyList = ["enterprise":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["enterprise":(["rocroller-ubuntu20-clang13":['rocroller-compile', 'rocroller-gfx90a'],
                                  "rocroller-ubuntu20-gcc":['rocroller-compile', 'rocroller-gfx90a']])]
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
            stage(jobName) {
                runCI(nodeDetails, jobName)
            }
    }

    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * 6')])]))
        stage(urlJobName) {
            runCI(["rocroller-ubuntu-clang13":['rocroller-compile']], urlJobName)
        }
    }
}
