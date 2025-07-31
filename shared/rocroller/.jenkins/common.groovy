// This file is for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

@NonCPS
def getPrComments(pullRequest) {
  ArrayList comments = pullRequest.comments.toList()
  return comments
}

def withSSH(platform, pipeline) {
    withCredentials(
        [
            sshUserPrivateKey(credentialsId:"github-rocmmathlibrariesbot-ssh_key-mathci_enterprise_job", keyFileVariable:"PUBLIC_KEY_FILE"),
            sshUserPrivateKey(credentialsId:"github_enterprise-a1_mlselibci_npi-ssh_key-mathci_enterprise_job", keyFileVariable: "ENTERPRISE_KEY_FILE"),
        ]
    )
    {
        configFileProvider(
            [configFile(fileId: 'github-enterprise-known-hosts', variable: 'ENTERPRISE_KNOWN_HOSTS'),
             configFile(fileId: 'github-enterprise-ssh-config', variable: 'ENTERPRISE_SSH_CONFIG')])
        {
            def sshBlock = """
            mkdir -p ~/.ssh/
            cat ${ENTERPRISE_KNOWN_HOSTS} >> ~/.ssh/known_hosts
            eval `ssh-agent -s`
            ssh-add ${PUBLIC_KEY_FILE}
            ssh-add ${ENTERPRISE_KEY_FILE}
            ssh-add -L
            cat ${ENTERPRISE_SSH_CONFIG} >> ~/.ssh/config
            """
            pipeline(sshBlock)
        }
    }
}

def runCompileCommand(platform, project, jobName, mxDataGeneratorGitURL, mxDataGeneratorGitTag, boolean codeCoverage=false, boolean enableTimers=false, String target='', boolean useYamlCpp=true, boolean staticAnalysis=false)
{
    project.paths.construct_build_prefix()
    String codeCovFlag = codeCoverage ? '-DROCROLLER_ENABLE_COVERAGE=ON -DROCROLLER_BUILD_SHARED_LIBS=OFF -DROCROLLER_ENABLE_LLD=ON' : ''
    String timerFlag = enableTimers ? '-DROCROLLER_ENABLE_TIMERS=ON' : ''
    String yamlBackendFlag = useYamlCpp ? '' : '-DROCROLLER_ENABLE_YAML_CPP=OFF'
    String useCppCheck = staticAnalysis ? '-DROCROLLER_ENABLE_CPPCHECK=ON' : ''

    mxDataGeneratorGitURL = mxDataGeneratorGitURL?.trim();
    mxDataGeneratorGitTag = mxDataGeneratorGitTag?.trim();
    String mxDataGeneratorGitURLFlag = mxDataGeneratorGitURL ? '-DMXDATAGENERATOR_GIT_URL=' + mxDataGeneratorGitURL : ''
    String mxDataGeneratorGitTagFlag = mxDataGeneratorGitTag ? '-DMXDATAGENERATOR_GIT_TAG=' + mxDataGeneratorGitTag : ''

    withSSH(platform) {
        sshBlock ->
        def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}

                ${sshBlock}

                mkdir -p build
                cd build
                # Check that all tests are included.
                ../scripts/check_included_tests.py
                cmake ../ \\
                    ${codeCovFlag} ${timerFlag} ${yamlBackendFlag} ${useCppCheck}\\
                    ${mxDataGeneratorGitURLFlag} ${mxDataGeneratorGitTagFlag}\\
                    -DCMAKE_CXX_COMPILER=/opt/rocm/bin/amdclang++ \\
                    -DCMAKE_BUILD_TYPE=Release \\
                    -DROCROLLER_ENABLE_FETCH=ON \\
                    -DCMAKE_PREFIX_PATH="/opt/rocm;/opt/rocm/llvm"
                ccache --print-stats
                make -j ${target}
                ccache --print-stats
                """

        platform.runCommand(this, command)
    }
}

def runTestCommand (platform, project)
{
    String testExclude = platform.jenkinsLabel.contains('compile') ? '-LE GPU' : ''

    def numThreads = 8

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}

                pushd build
                echo Using `nproc` threads for testing.
                OMP_NUM_THREADS=8 ctest -j ${numThreads} --output-on-failure ${testExclude}
                export ROCROLLER_BUILD_DIR="\$(pwd)"
                popd
                scripts/rrperf generate --suite generate_gfx950 --arch gfx950
            """

    try
    {
        platform.runCommand(this, command)
    }
    finally
    {
        junit "${project.paths.project_build_prefix}/build/test_report/**/*.xml"
    }
}

def runCodeCovTestCommand(platform, project)
{
    String masterURL = env.CHANGE_ID ? env.JOB_URL.replace("PR-${env.CHANGE_ID}", env.CHANGE_TARGET) : env.JOB_URL

    def compareCommand = """#!/usr/bin/env bash
                            set -ex

                            bash `pwd`/${project.paths.project_build_prefix}/scripts/codecov \\
                                -g ${platform.gpu} \\
                                -b `pwd`/${project.paths.project_build_prefix}/build \\
                                -u ${masterURL}/lastSuccessfulBuild/artifact/*zip*/archive.zip
                         """

    platform.runCommand(this, compareCommand)

    platform.archiveArtifacts(this, "${project.paths.project_build_prefix}/build/code_cov_${platform.gpu}.*")
    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/build/",
                reportFiles: "code_cov_${platform.gpu}_html/index.html,code_cov_diff_${platform.gpu}.html",
                reportName: "Code coverage ${platform.gpu} report",
                reportTitles: "Report,Diff"])

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/python_cov_html/",
                reportFiles: "index.html",
                reportName: "Python Code coverage ${platform.gpu} report",
                reportTitles: "Report"])

    if (env.CHANGE_ID)
    {
        def commentString = "# Code Coverage Report for ${platform.gpu}\n\n"
        def results = readFile("${project.paths.project_src_prefix}/build/code_cov_${platform.gpu}.formatted")
        def new_uncovered_lines = readFile("${project.paths.project_src_prefix}/build/new_uncovered_lines.txt").trim()
        commentString += "## Summary\n\n"
        commentString += "${results}\n\n"
        if ("${new_uncovered_lines}" != "0")
        {
            commentString += "**This PR adds/edits _${new_uncovered_lines}_ newly uncovered lines.**\n\n"
        }
        commentString += "## Artifacts\n\n"
        commentString += "* [HTML Coverage Report and Diff](${JOB_URL}/Code_20coverage_20${platform.gpu}_20report) \n"
        commentString += "* [File Coverage Summary](${JOB_URL}/lastSuccessfulBuild/artifact/${project.paths.src_prefix}/rocRoller/build/code_cov_${platform.gpu}.report/*view*/) \n"
        commentString += "* [Diff Text File](${JOB_URL}/lastSuccessfulBuild/artifact/${project.paths.src_prefix}/rocRoller/build/code_cov_${platform.gpu}.diff/*view*/) \n"
        commentString += "* [Full Text Coverage Report](${JOB_URL}/lastSuccessfulBuild/artifact/${project.paths.src_prefix}/rocRoller/build/code_cov_${platform.gpu}.zip) \n"
        commentString += "* [Python Coverage Report](${JOB_URL}/Python_20Code_20coverage_20${platform.gpu}_20report) \n"
        commentString += "\n"
        commentString += "## Commit Hashes\n\n"
        for(parentHash in project.gitParentHashes) {
            commentString += "* ${parentHash} \n"
        }
        boolean commentExists = false
        for (prComment in getPrComments(pullRequest)) {
            if (prComment.body.contains("# Code Coverage Report for ${platform.gpu}"))
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

def runBuildDocsCommand(platform, project)
{
    project.paths.construct_build_prefix()
    withSSH(platform) {
        sshBlock ->
        def command = """#!/usr/bin/env bash
                    set -ex
                    cd ${project.paths.project_build_prefix}
                    ${sshBlock}
                    cmake --preset docs -B build -S .
                    cmake --build build --target docs
                    """
        platform.runCommand(this, command)
    }

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}/docs/build/sphinx/html/",
                reportFiles: "index.html",
                reportName: "Generated Docs",
                reportTitles: "Docs"])

    if (env.CHANGE_ID)
    {
        def commentTitle = "# Generated Documentation"
        def commentString = "${commentTitle}\n\n"
        commentString += "* [Link to view Generated Docs.](${JOB_URL}/Generated_20Docs) \n\n"

        boolean commentExists = false
        for (prComment in getPrComments(pullRequest)) {
            if (prComment.body.contains(commentTitle))
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

def runPerformanceCommand (platform, project, mxDataGeneratorGitURL, mxDataGeneratorGitTag)
{
    String masterURL = env.CHANGE_ID ? env.JOB_URL.replace("PR-${env.CHANGE_ID}", env.CHANGE_TARGET) : env.JOB_URL

    mxDataGeneratorGitURL = mxDataGeneratorGitURL?.trim() ?: "";
    mxDataGeneratorGitTag = mxDataGeneratorGitTag?.trim() ?: "";

    withSSH(platform){
        sshBlock ->
        def rrperfSuite = platform.jenkinsLabel.contains('gfx12') ? "all_gfx120X" : "all"


        if (env.CHANGE_ID)
        {
            // either a label or a parameter can block comparison to master branch
            def masterCompare = !(
                pullRequest.labels.any { it == "ci:no-build-master" || it == "ci:no-build-target" }
            )
            if (masterCompare && (params?."Build target branch for comparison" != null))
            {
                masterCompare = params."Build target branch for comparison"
            }
            String masterCompareCommand
            if (masterCompare)
            {
                masterCompareCommand = """
                    export ROCROLLER_MXDATAGENERATOR_GIT_URL=${mxDataGeneratorGitURL}
                    export ROCROLLER_MXDATAGENERATOR_GIT_TAG=${mxDataGeneratorGitTag}
                    ./scripts/rrperf autoperf \\
                        --suite ${rrperfSuite} \\
                        --clonedir "./performance_build_${platform.gpu}" \\
                        --rundir "./performance_${platform.gpu}" \\
                        --plot_median --normalize \\
                        --x_value "commit" \\
                        --no-fail=remotes/origin/${env.CHANGE_TARGET} \\
                        "remotes/origin/${env.CHANGE_TARGET}"

                    mv ./performance_build_${platform.gpu}/**/comparison*.html \\
                        performance_comparison_${platform.gpu}.html
                    ./scripts/rrperf compare \\
                        \$(ls -trd ./performance_build_${platform.gpu}/performance_${platform.gpu}/*) \\
                            > performance_comparison_${platform.gpu}.md
                """
            }
            else
            {
                masterCompareCommand = """
                    export ROCROLLER_MXDATAGENERATOR_GIT_URL=${mxDataGeneratorGitURL}
                    export ROCROLLER_MXDATAGENERATOR_GIT_TAG=${mxDataGeneratorGitTag}
                    mkdir -p performance_build_${platform.gpu}
                    ./scripts/rrperf autoperf \\
                        --suite ${rrperfSuite} \\
                        --rundir "./performance_build_${platform.gpu}/performance_${platform.gpu}"
                    cat ./performance_build_${platform.gpu}/performance_${platform.gpu}/**/*.log >> performance_${platform.gpu}_logs.txt

                    #Get Master Results
                    wget ${masterURL}/lastSuccessfulBuild/artifact/*zip*/archive.zip
                    unzip archive.zip

                    if [ -f archive/*/*/performance_${platform.gpu}_last.zip ]; then

                        #Unzip last run's results
                        mv archive/*/*/performance_${platform.gpu}_last.zip ./performance_${platform.gpu}_master.zip
                        unzip performance_${platform.gpu}_master.zip -d ./performance_${platform.gpu}_master

                        # Clean up zip files and extracted contents
                        rm -rf archive.zip archive performance_${platform.gpu}_master.zip

                        #Compare Results to Master
                        ./scripts/rrperf compare \\
                            --plot_median --normalize \\
                            --x_value "commit" \\
                            --format "html" \\
                            ./performance_${platform.gpu}_master/performance_${platform.gpu}/* \\
                            ./performance_build_${platform.gpu}/performance_${platform.gpu}/* \\
                            > performance_comparison_${platform.gpu}.html

                        ./scripts/rrperf compare \\
                            ./performance_${platform.gpu}_master/performance_${platform.gpu}/* \\
                            ./performance_build_${platform.gpu}/performance_${platform.gpu}/* \\
                            > performance_comparison_${platform.gpu}.md
                    else
                        touch performance_comparison_${platform.gpu}.html
                        touch performance_comparison_${platform.gpu}.md
                        echo "Skipped ${env.CHANGE_TARGET} compare for ${platform.gpu}, no archived performance_${platform.gpu}_last.zip found."
                    fi
                """
            }
            def command = """#!/usr/bin/env bash
                        set -ex
                        cd ${project.paths.project_build_prefix}/

                        ${sshBlock}

                        #Run Performance Test
                        export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:${project.paths.project_build_prefix}/build/"
                        export ROCROLLER_BUILD_DIR="\$(pwd)/build"

                        ${masterCompareCommand}

                        #Zip archive
                        zip -r performance_${platform.gpu}_archive.zip \\
                            "./performance_build_${platform.gpu}/performance_${platform.gpu}"

                        # boost has files with UTF-8 characters that cannot be parsed by the HTML publisher
                        rm -rf build/_deps
                        rm -rf performance_build*/**/**/_deps
                    """
            platform.runCommand(this, command)

            platform.archiveArtifacts(this, "${project.paths.project_build_prefix}/performance_${platform.gpu}_archive.zip")

            publishHTML([allowMissing: false,
                        alwaysLinkToLastBuild: false,
                        keepAll: false,
                        reportDir: "${project.paths.project_build_prefix}/",
                        reportFiles: "performance_comparison_${platform.gpu}.html",
                        reportName: "Performance Report for ${platform.gpu}",
                        reportTitles: "Report"])

            def commentTitle = "# Performance Report for ${platform.gpu}"
            def commentString = "${commentTitle}\n\n"
            def results = readFile("${project.paths.project_build_prefix}/performance_comparison_${platform.gpu}.md").trim()
            def estimateString = masterCompare ? "" : " (estimated due to skipped ${env.CHANGE_TARGET} build)"
            commentString += "## Results${estimateString}\n\n"
            commentString += "${results}\n\n"
            commentString += "<details><summary>Links</summary>\n\n"
            commentString += "* [HTML Report](${JOB_URL}/Performance_20Report_20for_20${platform.gpu}) \n"
            commentString += "* [Job Link](${env.BUILD_URL}) \n"
            commentString += "* [Result Archive](${JOB_URL}/lastSuccessfulBuild/artifact/${project.paths.src_prefix}/rocRoller/performance_${platform.gpu}_archive.zip) \n"
            commentString += "</details>\n\n"

            boolean commentExists = false
            for (prComment in getPrComments(pullRequest)) {
                if (prComment.body.contains(commentTitle))
                {
                    commentExists = true
                    prComment.body = commentString
                }
            }
            if (!commentExists) {
                def comment = pullRequest.comment(commentString)
            }
        }
        else
        {
            def ARCHIVE_LIMIT = "101"

            def command = """#!/usr/bin/env bash
                        set -ex
                        cd ${project.paths.project_build_prefix}/

                        ${sshBlock}

                        #Get Master Results
                        wget ${masterURL}/lastSuccessfulBuild/artifact/*zip*/archive.zip
                        unzip archive.zip

                        #Run Performance Test
                        export LD_LIBRARY_PATH="\${LD_LIBRARY_PATH}:${project.paths.project_build_prefix}/build/"
                        export ROCROLLER_BUILD_DIR="\$(pwd)/build"
                        ./scripts/rrperf run \\
                            --suite ${rrperfSuite} \\
                            --rundir "./performance_${platform.gpu}"
                        cat ./performance_${platform.gpu}/**/*.log >> performance_${platform.gpu}_logs.txt

                        if [ -f archive/*/*/performance_${platform.gpu}_last.zip ]; then
                            #Unzip last run's results
                            mv archive/*/*/performance_${platform.gpu}_last.zip ./performance_${platform.gpu}_master.zip
                            unzip performance_${platform.gpu}_master.zip -d ./performance_${platform.gpu}_master

                            #Compare Results to Master
                            ./scripts/rrperf compare ./performance_${platform.gpu}_master/performance_${platform.gpu}/* ./performance_${platform.gpu}/* > performance_comparison_${platform.gpu}.md

                            #Make email report
                            ./scripts/rrperf compare --format email_html ./performance_${platform.gpu}_master/performance_${platform.gpu}/* ./performance_${platform.gpu}/* > performance_comparison_${platform.gpu}.email
                        else
                            touch performance_comparison_${platform.gpu}.md
                            touch performance_comparison_${platform.gpu}.email
                            echo "Skipped master compare for ${platform.gpu}, no archived performance_${platform.gpu}_last.zip found."
                        fi

                        #Zip current Results as last
                        zip -r performance_${platform.gpu}_last.zip "./performance_${platform.gpu}"

                        if [ -f archive/*/*/performance_${platform.gpu}_archive.zip ]; then
                            #Add past archive results for archiving
                            mv archive/*/*/performance_${platform.gpu}_archive.zip ./performance_${platform.gpu}_archive.zip
                            unzip ./performance_${platform.gpu}_archive.zip
                        else
                            echo "Skip archiving previous performance as none found, no performance_${platform.gpu}_archive.zip."
                        fi

                        #Only keep most recent results
                        ls -dr ./performance_${platform.gpu}/* | tail -n +${ARCHIVE_LIMIT} | xargs rm -rf

                        #Compare All Archived Results
                        ./scripts/rrperf compare --format html --group_results --y_zero --exclude_boxplot --plot_median ./performance_${platform.gpu}/* > performance_comparison_${platform.gpu}.html

                        #Zip archive
                        zip -r performance_${platform.gpu}_archive.zip "./performance_${platform.gpu}"

                        # boost has files with UTF-8 characters that cannot be parsed by the HTML publisher
                        rm -rf build/_deps
                        rm -rf performance_build*/**/**/_deps
                    """
            platform.runCommand(this, command)

            platform.archiveArtifacts(this, "${project.paths.project_build_prefix}/performance_${platform.gpu}_archive.zip")
            platform.archiveArtifacts(this, "${project.paths.project_build_prefix}/performance_${platform.gpu}_last.zip")

            publishHTML([allowMissing: false,
                        alwaysLinkToLastBuild: false,
                        keepAll: false,
                        reportDir: "${project.paths.project_build_prefix}/",
                        reportFiles: "performance_comparison_${platform.gpu}.html",
                        reportName: "Performance Report for ${platform.gpu}",
                        reportTitles: "Report"])

            def email_results = readFile("${project.paths.project_build_prefix}/performance_comparison_${platform.gpu}.email").trim()
            emailext (
                subject: "rocRoller Master Performance Results for ${platform.gpu}",
                body: """<h1>Performance Report for ${platform.gpu}</h1>
                        <h2>Links:</h2>
                        <ul>
                        <li><a href='${JOB_URL}/Performance_20Report_20for_20${platform.gpu}'>HTML Report</a></li>
                        <li><a href='${env.BUILD_URL}'>Job Link</a></li>
                        <li><a href='${JOB_URL}/lastSuccessfulBuild/artifact/${project.paths.src_prefix}/rocRoller/performance_${platform.gpu}_archive.zip'>Result Archive</a></li>
                        </ul>
                        ${email_results}""",
                to: "dl.rocroller@amd.com"
            )
        }
    }
}

def runCodeQLCompileCommand (platform, project, jobName, mxDataGeneratorGitURL, mxDataGeneratorGitTag)
{
    project.paths.construct_build_prefix()

    mxDataGeneratorGitURL = mxDataGeneratorGitURL?.trim() ?: "";
    mxDataGeneratorGitTag = mxDataGeneratorGitTag?.trim() ?: "";

    withSSH(platform) {
        sshBlock ->
        def command = """#!/usr/bin/env bash
                    set -ex
                    cd ${project.paths.project_build_prefix}

                    ${sshBlock}

                    export ROCROLLER_MXDATAGENERATOR_GIT_URL=${mxDataGeneratorGitURL}
                    export ROCROLLER_MXDATAGENERATOR_GIT_TAG=${mxDataGeneratorGitTag}
                    ./codeql/setup_codeql
                    ./codeql/create_database
                    """

        platform.runCommand(this, command)
    }
}

def runCodeQLTestCommand (platform, project)
{
    project.paths.construct_build_prefix()

    def command = """#!/usr/bin/env bash
                set -ex
                cd ${project.paths.project_build_prefix}

                # Run CodeQL unit tests
                ./codeql/run_tests

                ./codeql/analyze_database

                if [ ! -s "./codeql/build/codeql.csv" ]; then
                    echo "<h1>CodeQL Report</h1><h2>No errors to report!</h2>" > "./codeql/build/codeql.html"
                fi
                """

    platform.runCommand(this, command)

    platform.archiveArtifacts(this, "${project.paths.project_build_prefix}/codeql/build/codeql.html")
    platform.archiveArtifacts(this, "${project.paths.project_build_prefix}/codeql/build/codeql.sarif")
    platform.archiveArtifacts(this, "${project.paths.project_build_prefix}/codeql/build/types_count.md")

    publishHTML([allowMissing: false,
                alwaysLinkToLastBuild: false,
                keepAll: false,
                reportDir: "${project.paths.project_build_prefix}",
                reportFiles: "codeql/build/codeql.html",
                reportName: "CodeQL",
                reportTitles: "Report"])

    if (env.CHANGE_ID)
    {
        def commentTitle = "# CodeQL report"
        def commentString = "${commentTitle}\n\n"
        commentString += readFile("${project.paths.project_src_prefix}/codeql/build/types_count.md").trim() + "\n\n"
        commentString += "## Links\n"
        commentString += "* [HTML](${JOB_URL}CodeQL/codeql/build/codeql.html) \n"
        commentString += "* [Sarif](${JOB_URL}CodeQL/codeql/build/codeql.sarif) (for download and usage in conjunction with SARIF viewers)\n\n"

        boolean commentExists = false
        for (prComment in getPrComments(pullRequest)) {
            if (prComment.body.contains(commentTitle))
            {
                commentExists = true
                prComment.body = commentString
            }
        }
        if (!commentExists) {
            def comment = pullRequest.comment(commentString)
        }
    }

    def html_contents = readFile("${project.paths.project_src_prefix}/codeql/build/codeql.html")
    if(html_contents.contains("<td>error</td>"))
    {
        error('CodeQL report has errors!')
    }
}

return this
