#!/usr/bin/env groovy
// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, boolean debug=false, boolean buildStatic=false)
{
    def reference = (env.BRANCH_NAME ==~ /PR-\d+/) ? 'develop' : 'master'

    project.paths.construct_build_prefix()

    def getDependenciesCommand = ""
    if (project.installLibraryDependenciesFromCI)
    {
        project.libraryDependencies.each
        { libraryName ->
            getDependenciesCommand += auxiliary.getLibrary(libraryName, platform.jenkinsLabel, null, false)
        }
    }

    dir("${project.paths.project_build_prefix}/ref-repo") {
       git branch: "${reference}", url: 'https://github.com/ROCmSoftwarePlatform/rocFFT.git'
    }

    String clientArgs = '-DBUILD_CLIENTS_SAMPLES=ON -DBUILD_CLIENTS_TESTS=ON -DBUILD_CLIENTS_BENCH=ON'
    String noclientArgs = '-DBUILD_CLIENTS_SAMPLES=OFF -DBUILD_CLIENTS_TESTS=OFF -DBUILD_CLIENTS_BENCH=OFF'
    String warningArgs = '-DWERROR=ON'
    String buildTypeArg = debug ? '-DCMAKE_BUILD_TYPE=Debug -DROCFFT_DEVICE_FORCE_RELEASE=ON' : '-DCMAKE_BUILD_TYPE=Release'
    String buildTypeDir = debug ? 'debug' : 'release'
    String rtcBuildCache = "-DROCFFT_BUILD_KERNEL_CACHE_PATH=\$JENKINS_HOME_LOCAL/rocfft_build_cache.db"
    String cmake = platform.jenkinsLabel.contains('centos') ? 'cmake3' : 'cmake'

    def command = """#!/usr/bin/env bash
                set -x
                cd ${project.paths.project_build_prefix}
                ${getDependenciesCommand}
                set -e
                mkdir -p build/${buildTypeDir} && pushd build/${buildTypeDir}
                ${auxiliary.gfxTargetParser()}
                ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang -DAMDGPU_TARGETS=\$gfx_arch ${buildTypeArg} ${clientArgs} ${warningArgs} ${rtcBuildCache} ../..
                make -j\$(nproc)
                popd
                cd ref-repo
                mkdir -p build/${buildTypeDir} && pushd build/${buildTypeDir}
                ${auxiliary.gfxTargetParser()}
                ${cmake} -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ -DCMAKE_C_COMPILER=/opt/rocm/bin/amdclang -DAMDGPU_TARGETS=\$gfx_arch ${buildTypeArg} ${noclientArgs} ${warningArgs} ${rtcBuildCache} ../..
                make -j\$(nproc)
            """
    platform.runCommand(this, command)
}

def runTestCommand (platform, project, boolean debug=false)
{
    String sudo = auxiliary.sudo(platform.jenkinsLabel)
    String directory = debug ? 'debug' : 'release'

    def dataTypes = ['single', 'double']
    for (def dataType in dataTypes)
    {
        def command = """#!/usr/bin/env bash
                    set -ex
                    pwd
                    cd ${project.paths.project_build_prefix}
		    export ROCFFT_RTC_CACHE_PATH="\$JENKINS_HOME_LOCAL/rocfft_build_cache.db"
                    ./scripts/perf/rocfft-perf run --bench ./build/${directory}/clients/staging/dyna-rocfft-bench --lib ./ref-repo/build/${directory}/library/src/librocfft.so --lib ./build/${directory}/library/src/librocfft.so --out ./${dataType}_ref --out ./${dataType}_change --device 0 --precision ${dataType} --suite benchmarks
                    ls ${dataType}_change
                    ls ${dataType}_ref
                    mkdir ${dataType}_results
                    ./scripts/perf/rocfft-perf post ./${dataType}_results ./${dataType}_ref ./${dataType}_change
                    ls ${dataType}_change/*.mdat
                    ./scripts/perf/rocfft-perf html ./${dataType}_results ./${dataType}_ref ./${dataType}_change
                    mv ${dataType}_results/figs.html ${dataType}_results/figs_${platform.gpu}.html
                """
         platform.runCommand(this, command)

         archiveArtifacts "${project.paths.project_build_prefix}/${dataType}_results/*.html"
         publishHTML([allowMissing: false,
                    alwaysLinkToLastBuild: false,
                    keepAll: false,
                    reportDir: "${project.paths.project_build_prefix}/${dataType}_results",
                    reportFiles: "figs_${platform.gpu}.html",
                    reportName: "${dataType}-precision-${platform.gpu}",
                    reportTitles: "${dataType}-precision-${platform.gpu}"])
    }


    withCredentials([gitUsernamePassword(credentialsId: 'GitHub-ROCmMathLibrariesBot-Token', gitToolName: 'git-tool')])
    {
        platform.runCommand(
            this,
            """
            cd ${project.paths.build_prefix}
            git clone https://github.com/ROCmSoftwarePlatform/rocPTS.git -b release/rocpts-rel-1.2.0
            cd rocPTS
            python3 -m pip install build
            python3 -m build
            python3 -m pip install .
            """
        )
    }
    writeFile(
        file: project.paths.project_build_prefix + "/record_pts.py",
        text: libraryResource("com/amd/scripts/record_pts.py"))
    def setupBranch = env.CHANGE_ID ? "git branch \$BRANCH_NAME" : ""
    def command = """#!/usr/bin/env bash
    set -ex
    cd ${project.paths.project_build_prefix}
    ${setupBranch}
    git checkout \$BRANCH_NAME
    benchmark_folder=rocFFT_Benchmark_Dataset_\$(date +%Y%m%d)
    mkdir -p \${benchmark_folder}/all_change \${benchmark_folder}/all_ref
    cp -uf ./*_change/* \${benchmark_folder}/all_change
    cp -uf ./*_ref/* \${benchmark_folder}/all_ref
    python3 ./record_pts.py \
        --dataset-path \$PWD/\${benchmark_folder} \
        --reference-dataset all_ref \
        --new-dataset all_change \
        --new-build . \
        --reference-build ./ref-repo\
        -v 5.5 \
        -l pts_rocfft_benchmark_data-v1.0.0
    """
    withCredentials([usernamePassword(credentialsId: 'PTS_API_ID_KEY_PROD', usernameVariable: 'PTS_API_ID', passwordVariable: 'PTS_API_KEY')])
    {
        platform.runCommand(this, command)
    }
}

def runCI =
{
    nodeDetails, jobName->

    def prj  = new rocProject('rocFFT-internal', 'Performance')

    prj.defaults.ccache = true
    prj.timeout.compile = 600
    prj.timeout.test = 600
    prj.libraryDependencies = ['rocRAND','hipRAND']

    // Define test architectures, optional rocm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false

    def commonGroovy
    def gpus = []
    def dataTypes = ['single', 'double']

    def compileCommand =
    {
        platform, project->

        gpus.add(platform.gpu)
        commonGroovy = load "${project.paths.project_src_prefix}/.jenkins/common.groovy"
        runCompileCommand(platform, project, jobName)
    }

    def testCommand =
    {
        platform, project->

        runTestCommand(platform, project)
    }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, testCommand, null)
    def commentString = "Performance reports: \n" + "Commit hashes: \n"
    for (parentHash in prj.gitParentHashes) {
         commentString += "${parentHash} \n"
    }
    for (gpu in gpus) {
        for (dataType in dataTypes) {
            commentString += "[${gpu} ${dataType} report](${JOB_URL}/${dataType}-precision-${gpu})\n"
        }
    }

    if (env.BRANCH_NAME ==~ /PR-\d+/)
    {
        boolean commentExists = false
        for (prComment in pullRequest.comments) {
            if (prComment.body.contains("Performance reports:"))
            {
                commentExists = true
                prComment.body = commentString
            }
        }
        if (!commentExists) {
            def comment = pullRequest.comment(commentString)
        }
    }
}

ci: {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    def propertyList = ["compute-rocm-dkms-no-npi-hipclang":[pipelineTriggers([cron('0 1 * * 0')])]]
    propertyList = auxiliary.appendPropertyList(propertyList)

    def jobNameList = ["compute-rocm-dkms-no-npi-hipclang":([ubuntu18:['gfx900','gfx906']])]
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

    // For url job names that are not listed by the jobNameList i.e. compute-rocm-dkms-no-npi-1901
    if(!jobNameList.keySet().contains(urlJobName))
    {
        properties(auxiliary.addCommonProperties([pipelineTriggers([cron('0 1 * * *')])]))
        stage(urlJobName) {
            runCI([ubuntu18:['gfx906']], urlJobName)
        }
    }
}
