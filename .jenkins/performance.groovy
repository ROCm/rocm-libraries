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
    nodeDetails, jobName->

    def prj = new rocProject('rocRoller', 'Performance')

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
        commonGroovy.runCompileCommand(platform, project, jobName, false, true, 'all_clients', true) //TODO: Switch last arg back to false after fixing YAML_BACKEND=LLVM
    }

    def testCommand =
    {
        platform, project->

        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        commonGroovy.runPerformanceCommand(platform, project)
    }

    if (env.CHANGE_ID)
    {
        buildProject(prj, formatCheck, nodes.dockerArray, null, testCommand, null)
    }
    else
    {
        buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
    }
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["enterprise":[pipelineTriggers([cron('0 H(0-5) * * *')])]]
    def additionalParameters = [
        string(name: "AMDGPU_URL", defaultValue: params?.AMDGPU_URL ?: "", description: "URL to retrieve AMDGPU install package from"),
        string(name: "AMDGPU_BUILD_NUMBER", defaultValue: params?.AMDGPU_BUILD_NUMBER ?: "", description: "Build number to use for AMDGPU"),
        string(name: "AMDGPU_BUILD_URI", defaultValue: params?.AMDGPU_BUILD_URI ?: "", description: "Specify the specific artifact path for AMDGPU")
    ]

    if(env.CHANGE_ID){
        propertyList = ["enterprise":[pipelineTriggers([cron('0 1 * * 0')])]]
        additionalParameters += [
            booleanParam(
                name: "Build master branch for comparison",
                defaultValue: true,
                description: "Clone and build the master branch for performance " +
                    "comparison (if unchecked, will compare to latest results " +
                    "from master branch)"
            )
        ]
    }

    auxiliary.registerAdditionalParameters(additionalParameters)
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
