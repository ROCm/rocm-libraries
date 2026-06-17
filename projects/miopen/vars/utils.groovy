/*******************************************************************************
 *
 * MIT License
 *
 * Copyright (c) 2025 Advanced Micro Devices, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

import org.jenkinsci.plugins.workflow.graph.FlowGraphWalker
import org.jenkinsci.plugins.workflow.actions.ErrorAction
import org.jenkinsci.plugins.workflow.actions.LabelAction
import org.jenkinsci.plugins.workflow.actions.ThreadNameAction
import org.jenkinsci.plugins.workflow.cps.nodes.StepStartNode
import org.jenkinsci.plugins.workflow.cps.nodes.StepEndNode

def miopenCheckout()
{
    // checkout project
    checkout([
        $class: 'GitSCM',
        branches: scm.branches,
        doGenerateSubmoduleConfigurations: scm.doGenerateSubmoduleConfigurations,
        extensions: scm.extensions + [[$class: 'CleanCheckout']] + [[$class: 'SubmoduleOption', disableSubmodules: false, parentCredentials: true, recursiveSubmodules: true, ]],
        userRemoteConfigs: scm.userRemoteConfigs
    ])
}

def check_host() {
    if ("${env.MIOPEN_SCCACHE}" != "null"){
        def SCCACHE_SERVER="${env.MIOPEN_SCCACHE.split(':')[0]}"
        echo "sccache server: ${SCCACHE_SERVER}"
        sh '''ping -c 1 -p 6379 "${SCCACHE_SERVER}" | echo $? > tmp.txt'''
        def output = readFile(file: "tmp.txt")
        echo "tmp.txt contents: \$output"
        return (output != "0")
    }
    else{
        return 1
    }
}

//default
// CXX=/opt/rocm/llvm/bin/clang++ CXXFLAGS='-Werror' cmake -G Ninja -DMIOPEN_GPU_SYNC=Off -DCMAKE_PREFIX_PATH=/usr/local -DBUILD_DEV=On -DCMAKE_BUILD_TYPE=release ..
//
def cmake_build(Map conf=[:]){

    def compiler = conf.get("compiler","/opt/rocm/llvm/bin/clang++")
    def make_targets = conf.get("make_targets","check")
    def debug_flags = "-g -fno-omit-frame-pointer -fsanitize=undefined -fno-sanitize-recover=undefined -Wno-option-ignored " + conf.get("extradebugflags", "")
    def build_envs = "CTEST_PARALLEL_LEVEL=4 " + conf.get("build_env","")
    def prefixpath = conf.get("prefixpath","/opt/rocm")
    def build_type_debug = (conf.get("build_type",'release') == 'debug')
    def miopen_install_path = conf.get("miopen_install_path", "${env.WORKSPACE}/${env.MIOPEN_DIR}/install")

    def mlir_args = " -DMIOPEN_USE_MLIR=" + conf.get("mlir_build", "ON")
    // WORKAROUND_ISSUE_3192 Disabling MLIR for debug builds since MLIR generates sanitizer errors.
    if (build_type_debug)
    {
        mlir_args = " -DMIOPEN_USE_MLIR=OFF"
    }

    def setup_args = mlir_args + " -DMIOPEN_GPU_SYNC=Off " + conf.get("setup_flags","")
    def build_fin = conf.get("build_fin", "OFF")

    setup_args = setup_args + " -DCMAKE_PREFIX_PATH=${prefixpath} "

    //cmake_env can overwrite default CXX variables.
    def cmake_envs = "CXX=${compiler} CXXFLAGS='-Werror' " + conf.get("cmake_ex_env","")

    def package_build = (conf.get("package_build",false) == true)

    if (package_build == true) {
        make_targets = "miopen_gtest package miopen_gtest_check"
        setup_args = " -DMIOPEN_TEST_DISCRETE=OFF " + setup_args
    }

    if(conf.get("build_install",false) == true)
    {
        make_targets = 'install ' + make_targets
        setup_args = " -DBUILD_DEV=Off -DCMAKE_INSTALL_PREFIX=${miopen_install_path}" + setup_args
    } else if(package_build == true) {
        setup_args = ' -DBUILD_DEV=Off' + setup_args
    } else {
        setup_args = ' -DBUILD_DEV=On' + setup_args
    }

    // test_flags = ctest -> MIopen flags
    def test_flags = conf.get("test_flags","")

    if (conf.get("vcache_enable","") == "true"){
        //grab root of node workspace. not guaranteed to be /var/jenkins
        String remote_root = env.WORKSPACE.substring(0, env.WORKSPACE.lastIndexOf("workspace/"))
        def vcache = conf.get(vcache_path,"${remote_root}/.cache/miopen/vcache")
        build_envs = " MIOPEN_VERIFY_CACHE_PATH='${vcache}' " + build_envs
    } else{
        test_flags = " --disable-verification-cache " + test_flags
    }

    if(build_type_debug){
        setup_args = " -DCMAKE_BUILD_TYPE=debug -DCMAKE_CXX_FLAGS_DEBUG='${debug_flags}'" + setup_args
    }else{
        setup_args = " -DCMAKE_BUILD_TYPE=release" + setup_args
    }

    if(test_flags != ""){
       setup_args = "-DMIOPEN_TEST_FLAGS='${test_flags}'" + setup_args
    }

    if(conf.containsKey("find_mode"))
    {
        def fmode = conf.get("find_mode", "")
        setup_args = " -DMIOPEN_DEFAULT_FIND_MODE=${fmode} " + setup_args
    }
    if(env.CCACHE_HOST)
    {
        setup_args = " -DCMAKE_CXX_COMPILER_LAUNCHER='ccache' -DCMAKE_C_COMPILER_LAUNCHER='ccache' " + setup_args
    }

    if ( build_fin == "ON" )
    {
        setup_args = " -DMIOPEN_INSTALL_CXX_HEADERS=On " + setup_args
    }

    def pre_setup_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            cd ${env.WORKSPACE}/${env.MIOPEN_DIR}
            rm -rf build
            mkdir build
            rm -rf install
            mkdir install
            rm -f src/kernels/*.ufdb.txt
            rm -f src/kernels/miopen*.udb
            cd build
        """
    def setup_cmd = conf.get("setup_cmd", "${cmake_envs} cmake -G Ninja ${setup_args}   .. ")
    // WORKAROUND_SWDEV_290754
    // It seems like this W/A is not required since 4.5.
    def build_cmd = conf.get("build_cmd", "LLVM_PATH=/opt/rocm/llvm ${build_envs} dumb-init ninja -j\$(nproc) ${make_targets}")
    def execute_cmd = conf.get("execute_cmd", "")

    def cmd = conf.get("cmd", """
            ${pre_setup_cmd}
            ${setup_cmd}
            ${build_cmd}
        """)

    if ( build_fin == "ON" )
    {
        def fin_build_cmd = cmake_fin_build_cmd(miopen_install_path)
        cmd += """
            export RETDIR=\$PWD
            cd ${env.WORKSPACE}/${env.MIOPEN_DIR}/fin
            ${fin_build_cmd}
            cd \$RETDIR
        """
    }

    cmd += """
        ${execute_cmd}
    """

    echo cmd
    sh cmd

    // Only archive from master or develop
    if (package_build == true && (env.BRANCH_NAME == "develop" || env.BRANCH_NAME == "master" ||
        params.PERF_TEST_ARCHIVE == true)) {
        archiveArtifacts artifacts: "build/*.deb", allowEmptyArchive: true, fingerprint: true
        archiveArtifacts artifacts: "build/*.rpm", allowEmptyArchive: true, fingerprint: true
        stash includes: "build/*tar.gz", name: 'miopen_tar'
    }
}

def cmake_fin_build_cmd(prefixpath){
    def flags = "-DCMAKE_INSTALL_PREFIX=${prefixpath} -DCMAKE_BUILD_TYPE=release"
    def compiler = 'clang++'
    def make_targets = "install"
    def compilerpath = "/opt/rocm/llvm/bin/" + compiler
    def configargs = ""
    if (prefixpath != "")
    {
        configargs = "-DCMAKE_PREFIX_PATH=${prefixpath}"
    }

    def fin_cmd = """
            echo \$HSA_ENABLE_SDMA
            ulimit -c unlimited
            rm -rf build
            mkdir build
            cd build
            CXX=${compilerpath} cmake ${configargs} ${flags} ..
            dumb-init make -j\$(nproc) ${make_targets}
    """
    return fin_cmd
}

def getDockerImageName(dockerArgs)
{
    sh "echo ${dockerArgs} > ${env.WORKSPACE}/factors.txt"
    // Include the candidate image so the CI docker hash changes per TheRock hash.
    if (env.THEROCK_CANDIDATE_IMAGE) {
        sh "echo ${env.THEROCK_CANDIDATE_IMAGE} >> ${env.WORKSPACE}/factors.txt"
    }
    def image = "${env.MIOPEN_DOCKER_IMAGE_URL}"
    // Note: The following files and directories from the CK repo are used to generate a hash for
    // the docker image build. To ensure that we rebuild the docker image only when necessary.
    // Add any other files or directories that should trigger a rebuild of the docker image when changed.
    sh """
        cd ${env.WORKSPACE}/${env.CK_DIR} && \
        { \
            find cmake experimental include library -type f -print0; \
            find . -maxdepth 1 -type f \\( \
                -name 'CMakeLists.txt' -o \
                -name 'Config.cmake.in' -o \
                -name 'dev-requirements.txt' -o \
                -name 'pyproject.toml' -o \
                -name 'rbuild.ini' -o \
                -name 'requirements.txt' \
            \\) -print0; \
        } \
        | tr '\\0' '\\n' \
        | LC_ALL=C sort \
        | xargs -d '\\n' md5sum \
        | LC_ALL=C sort \
        | md5sum \
        | awk '{print \$1}' >> "${env.WORKSPACE}/factors.txt"
    """

    sh "cd ${env.WORKSPACE}/${env.MIOPEN_DIR}/ && md5sum Dockerfile requirements.txt dev-requirements.txt >> ${env.WORKSPACE}/factors.txt"
    def docker_hash = sh(script: "cd ${env.WORKSPACE} && md5sum factors.txt | awk '{print \$1}' | head -c 6", returnStdout: true)
    sh "rm ${env.WORKSPACE}/factors.txt"
    echo "Docker tag hash: ${docker_hash}"
    image = "${image}:ci_${docker_hash}"
    if(params.DOCKER_IMAGE_OVERRIDE && !params.DOCKER_IMAGE_OVERRIDE.empty)
    {
        echo "Overriding the base docker image with ${params.DOCKER_IMAGE_OVERRIDE}"
        image = "${params.DOCKER_IMAGE_OVERRIDE}"
    }
    return image
}

// Builds rocm/miopen:therock-<shortHash> from source; returns {image, fullHash, shortHash, skip}.
// Skips if :therock already carries this hash; reuses the hash-tagged image if it exists.
def buildTheRockDockerImage(Map conf=[:])
{
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm")

    def cacheRef = "${env.MIOPEN_DOCKER_IMAGE_URL}-ci-docker:therock_cache"

    def gpu_arch = "gfx908;gfx90a;gfx942;gfx950;gfx1101;gfx1151;gfx1201" // multiarch builds

    // Read the TheRock hash from the ci-env action (single source of truth).
    def theRockHash = sh(
        script: """
            grep -A 2 'therock-ref:' ${env.WORKSPACE}/.github/actions/ci-env/action.yml \
            | grep 'value:' \
            | awk '{print \$2}' \
            | tr -d '"'
        """.stripIndent(),
        returnStdout: true
    ).trim()

    def shortHash   = theRockHash.take(7)
    def hashedImage = "${env.MIOPEN_DOCKER_IMAGE_URL}:therock-${shortHash}"

    // Check whether this hash is already live on :therock via its baked-in label.
    // Pull and inspect are separated so that docker pull stdout does not contaminate the captured label.
    def lastPromotedHash = ""
    try {
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            sh "docker pull ${env.MIOPEN_DOCKER_IMAGE_URL}:therock > /dev/null 2>&1 || true"
            lastPromotedHash = sh(
                script: """
                    docker inspect \
                        --format '{{ index .Config.Labels "therock.git.hash" }}' \
                        ${env.MIOPEN_DOCKER_IMAGE_URL}:therock 2>/dev/null || true
                """.stripIndent(),
                returnStdout: true
            ).trim()
        }
    } catch (Exception e) {
        echo "Could not read label from existing :therock image (first-time run?): ${e.message}"
    }

    if (lastPromotedHash == theRockHash) {
        echo "TheRock hash ${shortHash} is already promoted to :therock - skipping build."
        return [image: null, fullHash: theRockHash, shortHash: shortHash, skip: true]
    }
    echo "New TheRock hash detected: ${theRockHash} (previously promoted: '${lastPromotedHash ?: 'none'}')"

    // Reuse the hash-tagged image if a previous nightly already built it.
    def imageAlreadyBuilt = false
    withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
        def rc = sh(script: "docker manifest inspect ${hashedImage} > /dev/null 2>&1", returnStatus: true)
        imageAlreadyBuilt = (rc == 0)
    }
    if (imageAlreadyBuilt) {
        echo "Hash-tagged image ${hashedImage} already exists - reusing without rebuild."
    } else {
        echo "Hash-tagged image ${hashedImage} not found - will build now."
    }

    if (!imageAlreadyBuilt) {
        def dockerArgs = "--build-arg PREFIX=${prefixpath} " +
                         "--build-arg THEROCK_GIT_HASH=\"${theRockHash}\" " +
                         "--build-arg THEROCK_ASIC=\"${gpu_arch}\" " +
                         "--build-arg BUILD_TYPE=build " +
                         "--label therock.git.hash=${theRockHash} " +
                         "--target update_therock " +
                         " -f ${env.WORKSPACE}/${env.MIOPEN_DIR}/Dockerfile "

        if (params.USE_SCCACHE_DOCKER && check_host() && "${env.MIOPEN_SCCACHE}" != "null") {
            dockerArgs = dockerArgs + " --build-arg MIOPEN_SCCACHE=${env.MIOPEN_SCCACHE} --build-arg COMPILER_LAUNCHER=sccache "
        }

        echo "Building ${hashedImage} with args: ${dockerArgs}"

        def buildContext    = "${env.WORKSPACE}/${env.PROJ_DIR}/."
        def dockerCacheArgs = "--cache-to type=registry,ref=${cacheRef},compression=zstd,mode=min " +
                              "--cache-from type=registry,ref=${cacheRef} "

        try {
            withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                sh """
                    docker buildx inspect ci-builder >/dev/null 2>&1 || \
                    docker buildx create --name ci-builder --driver docker-container --use
                    docker buildx use ci-builder
                    docker buildx inspect --bootstrap
                """.stripIndent()

                sh """
                    DOCKER_BUILDKIT=1 docker buildx build \
                    --push \
                    --tag ${hashedImage} \
                    ${dockerCacheArgs} \
                    ${dockerArgs} \
                    ${buildContext}
                """.stripIndent()
            }
        } catch (Exception bex) {
            echo "Buildx not available or failed, falling back to docker.build"
            def dockerImage = docker.build("${hashedImage}", "${dockerArgs} ${buildContext}")
            withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                dockerImage.push()
            }
        }
    }

    return [image: hashedImage, fullHash: theRockHash, shortHash: shortHash, skip: false]
}

// Retags the CI image as rocm/miopen-dev:multiarch_dev_<date> and :latest.
// Uses CI_DOCKER_IMAGE set by the Build Docker stage to avoid recomputing the image name.
def publishDevDockerImage(Map conf=[:])
{
    def date = new Date().format('yyyyMMdd')
    def devImageUrl = "${env.MIOPEN_DOCKER_IMAGE_URL}-dev"
    def dateTag   = "${devImageUrl}:multiarch_dev_${date}"
    def latestTag = "${devImageUrl}:latest"
    def ciImage   = env.CI_DOCKER_IMAGE

    if (!ciImage) {
        error "CI_DOCKER_IMAGE is not set - Build Docker stage must run before Publish Dev Image."
    }

    echo "Publishing dev image: ${ciImage} -> ${dateTag}"
    withDockerRegistry([credentialsId: "docker_test_cred", url: ""]) {
        sh """
            docker pull ${ciImage}
            docker tag  ${ciImage} ${dateTag}
            docker tag  ${ciImage} ${latestTag}
            docker push ${dateTag}
            docker push ${latestTag}
        """.stripIndent()
    }
}

// Re-tags the hash-tagged image as :therock; the baked label is preserved by docker tag.
def promoteTheRockDockerImage(String hashedImage, String fullHash)
{
    def targetImage = "${env.MIOPEN_DOCKER_IMAGE_URL}:therock"
    echo "Promoting ${hashedImage} -> ${targetImage} (hash: ${fullHash})"
    withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
        sh """
            docker pull ${hashedImage}
            docker tag  ${hashedImage} ${targetImage}
            docker push ${targetImage}
        """.stripIndent()
    }
    echo "Promotion complete - :therock now points to TheRock hash ${fullHash}"
}


// Embeds TheRock and CK git hashes as Docker image labels.
private def embedBuildMetadata(String dockerArgs) {
    try {
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            sh "docker pull ${env.MIOPEN_DOCKER_IMAGE_URL}:therock > /dev/null 2>&1 || true"
            def promotedHash = sh(
                script: """
                    docker inspect --format '{{ index .Config.Labels "therock.git.hash" }}' \
                        ${env.MIOPEN_DOCKER_IMAGE_URL}:therock 2>/dev/null || true
                """.stripIndent(),
                returnStdout: true
            ).trim()
            if (promotedHash) {
                echo "Embedding TheRock hash into CI image metadata: ${promotedHash}"
                dockerArgs = dockerArgs + "--label therock.git.hash=${promotedHash} "
                env.THEROCK_PROMOTED_HASH = promotedHash
            }
        }
    } catch (Exception e) {
        echo "Could not read TheRock label from :therock image, skipping metadata embedding: ${e.message}"
    }

    def ckHash = sh(
        script: "git -C ${env.WORKSPACE}/${env.CK_DIR} rev-parse HEAD",
        returnStdout: true
    ).trim()
    if (ckHash) {
        echo "Embedding CK hash into CI image metadata: ${ckHash}"
        dockerArgs = dockerArgs + "--label ck.git.hash=${ckHash} "
        env.CK_GIT_HASH = ckHash
    }
    return dockerArgs
}


def getDockerImage(Map conf=[:])
{
    env.DOCKER_BUILDKIT=1
    def prefixpath = conf.get("prefixpath", "/opt/rocm") // one image for each prefix 1: /usr/local 2:/opt/rocm

    def gpu_family = conf.get("gpu_family")

    def cacheRef = "${env.MIOPEN_DOCKER_IMAGE_URL}-ci-docker:cache_${gpu_family}"

    // Note: With offload compress disabled for CK expanding the target list might cause issues with the docker build.
    def gpu_arch
    if (gpu_family == "ci")
    {
        gpu_arch = "gfx908;gfx90a;gfx942;gfx950;gfx1101;gfx1151" // Builds docker image with subset of architectures that CI is run on.
    }
    else if (gpu_family == "gfx90X")
    {
        gpu_arch = "gfx908;gfx90a"
    }
    else if (gpu_family == "gfx942")
    {
        gpu_arch = "gfx942"
    }
    else if (gpu_family == "gfx950")
    {
        gpu_arch = "gfx950"
    }
    else if (gpu_family == "gfx942_gfx950")
    {
        gpu_arch = "gfx942;gfx950"
    }
    else if (gpu_family == "navi")
    {
        gpu_arch = "gfx1101;gfx1151"
    }
    else
    {
        error("Unsupported GPU family: ${gpu_family}")
    }

    def dockerArgs = "--build-arg PREFIX=${prefixpath} " +
                     "--target miopen "

    // Build the CI image FROM the candidate TheRock base when one is staged.
    if (env.THEROCK_CANDIDATE_IMAGE) {
        dockerArgs = dockerArgs + "--build-arg THEROCK_BASE_IMAGE=${env.THEROCK_CANDIDATE_IMAGE} "
    }

    if(env.CCACHE_HOST)
    {
        def check_host = sh(script:"""(printf "PING\r\n";) | nc -N ${env.CCACHE_HOST} 6379 """, returnStdout: true).trim()
        if(check_host == "+PONG")
        {
            echo "FOUND CCACHE SERVER: ${CCACHE_HOST}"
        }
        else
        {
            echo "CCACHE SERVER: ${CCACHE_HOST} NOT FOUND, got ${check_host} response"
        }
        dockerArgs = dockerArgs + " --build-arg CCACHE_SECONDARY_STORAGE='redis://${env.CCACHE_HOST}' --build-arg COMPILER_LAUNCHER='ccache' "
        env.CCACHE_DIR = """/tmp/ccache_store"""
        env.CCACHE_SECONDARY_STORAGE="""redis://${env.CCACHE_HOST}"""
    }
    else if (params.USE_SCCACHE_DOCKER && check_host() && "${env.MIOPEN_SCCACHE}" != "null")
    {
        dockerArgs = dockerArgs + " --build-arg MIOPEN_SCCACHE=${env.MIOPEN_SCCACHE} --build-arg COMPILER_LAUNCHER=sccache "
    }

    def image = getDockerImageName(dockerArgs)

    // Do not append gpu family for common ci image
    if(gpu_family != "ci"){
        image = image + "_${gpu_family}"
    }

    // Append GPU arch after image name for a common hash
    dockerArgs = dockerArgs + "--build-arg THEROCK_ASIC=\"${gpu_arch}\" "

    // Append Dockerfile path after image name is generated to avoid affecting the hash.
    dockerArgs = dockerArgs + " -f ${env.WORKSPACE}/${env.MIOPEN_DIR}/Dockerfile "

    // ensure_only: true = check registry only (manifest inspect), build if absent.
    // ensure_only: false (default) = pull to local daemon, build if pull fails.
    def ensure_only = conf.get("ensure_only", false)

    def dockerImage
    if (ensure_only) {
        def remoteExists = withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            sh(script: "docker manifest inspect ${image} > /dev/null 2>&1", returnStatus: true) == 0
        }
        if (remoteExists) {
            echo "Image ${image} already exists in registry - skipping build."
            dockerImage = docker.image("${image}")
        } else {
            dockerArgs = embedBuildMetadata(dockerArgs)
            echo "Docker Args: ${dockerArgs}"
            echo "Building image..."
            def buildContext = "${env.WORKSPACE}/${env.PROJ_DIR}/."
            def dockerCacheArgs = "--cache-to type=registry,ref=${cacheRef},compression=zstd,mode=max,registry.insecure=true " +
                                  "--cache-from type=registry,ref=${cacheRef},registry.insecure=true "
            try {
                withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                    sh """
                        docker buildx rm ci-builder || true
                        docker buildx create --name ci-builder --driver docker-container --use
                        docker buildx use ci-builder
                        docker buildx inspect --bootstrap
                    """.stripIndent()
                    sh """
                        DOCKER_BUILDKIT=1 docker buildx build \
                        --builder ci-builder \
                        --push \
                        --tag ${image} \
                        ${dockerCacheArgs} \
                        ${dockerArgs} \
                        ${buildContext}
                    """.stripIndent()
                }
                dockerImage = docker.image("${image}")
            } catch (Exception bex) {
                echo "Buildx not available or failed, falling back to docker.build"
                dockerImage = docker.build("${image}", "${dockerArgs} ${buildContext}")
                withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                    dockerImage.push()
                }
            }
        }
    } else {
        try{
            echo "Pulling down image: ${image}"
            dockerImage = docker.image("${image}")
            withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                dockerImage.pull()
            }
            def embeddedTheRockHash = sh(
                script: "docker inspect --format '{{ index .Config.Labels \"therock.git.hash\" }}' ${image} 2>/dev/null || true",
                returnStdout: true
            ).trim()
            def embeddedCkHash = sh(
                script: "docker inspect --format '{{ index .Config.Labels \"ck.git.hash\" }}' ${image} 2>/dev/null || true",
                returnStdout: true
            ).trim()
            echo "CI image TheRock hash: ${embeddedTheRockHash ?: 'not set'} | CK hash: ${embeddedCkHash ?: 'not set'}"
        }
        catch(org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
            echo "The job was cancelled or aborted"
            throw e
        }
        catch(Exception ex)
        {
            dockerArgs = embedBuildMetadata(dockerArgs)
            echo "Docker Args: ${dockerArgs}"
            echo "Building image..."
            def buildContext = "${env.WORKSPACE}/${env.PROJ_DIR}/."
            def dockerCacheArgs = "--cache-to type=registry,ref=${cacheRef},compression=zstd,mode=max,registry.insecure=true " +
                                  "--cache-from type=registry,ref=${cacheRef},registry.insecure=true "

            try {
                withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                    sh """
                        docker buildx rm ci-builder || true
                        docker buildx create --name ci-builder --driver docker-container --use
                        docker buildx use ci-builder
                        docker buildx inspect --bootstrap
                    """.stripIndent()
                    sh """
                        DOCKER_BUILDKIT=1 docker buildx build \
                        --builder ci-builder \
                        --push \
                        --tag ${image} \
                        ${dockerCacheArgs} \
                        ${dockerArgs} \
                        ${buildContext}
                    """.stripIndent()
                }
                dockerImage = docker.image("${image}")
            } catch (Exception bex) {
                echo "Buildx not available or failed, falling back to docker.build"
                dockerImage = docker.build("${image}", "${dockerArgs} ${buildContext}")
                withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                    dockerImage.push()
                }
            }
        }
    }

    if(params.INSTALL_MIOPEN == 'ON')
    {
        def freckle = sh(script: 'git rev-parse --short HEAD', returnStdout: true).trim()
        dockerArgs = " --build-arg BASE_DOCKER=${image} --build-arg FRECKLE=${freckle} -f ${env.WORKSPACE}/${env.MIOPEN_DIR}/Dockerfile.perftests"

        // Get updated image name for perf tests.
        image = getDockerImageName(dockerArgs)
        image = image + "_perfTest"

        if (ensure_only) {
            def remotePerfExists = withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                sh(script: "docker manifest inspect ${image} > /dev/null 2>&1", returnStatus: true) == 0
            }
            if (remotePerfExists) {
                echo "Perf test image ${image} already exists in registry - skipping build."
                dockerImage = docker.image("${image}")
            } else {
                dockerImage = docker.build("${image}", "${dockerArgs} -f ${env.WORKSPACE}/${env.MIOPEN_DIR}/Dockerfile ")
                withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                    dockerImage.push()
                }
            }
        } else {
            try{
                echo "Pulling down perf test image: ${image}"
                dockerImage = docker.image("${image}")
                withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                    dockerImage.pull()
                }
            }
            catch(org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                throw e
            }
            catch(Exception ex)
            {
                dockerImage = docker.build("${image}", "${dockerArgs} -f ${env.WORKSPACE}/${env.MIOPEN_DIR}/Dockerfile ")
                withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
                    dockerImage.push()
                }
            }
        }
    }

    return [dockerImage, image]
}

def setGithubStatus(String context, String state, String description) {
    def sha = env.GIT_COMMIT
    def targetUrl = env.RUN_DISPLAY_URL ?: env.BUILD_URL
    def statusUrl = "https://api.github.com/repos/ROCm/rocm-libraries/statuses/${sha}"
    withCredentials([usernamePassword(credentialsId: 'github-app-miopen', usernameVariable: 'GITHUB_APP', passwordVariable: 'GITHUB_TOKEN')]) {
        def code = '0'
        try {
            retry(3) {
                code = sh(returnStdout: true, script: """
                    curl -s -w "%{http_code}" -o /dev/null -X POST '${statusUrl}' \\
                        -H "Authorization: token \$GITHUB_TOKEN" \\
                        -H 'Content-Type: application/json' \\
                        -d '{"state":"${state}","context":"${context}","description":"${description}","target_url":"${targetUrl}"}'
                """).trim()
                if (!code.startsWith('2')) {
                    error("GitHub status POST returned ${code}")
                }
            }
        } catch (Exception e) {
            echo "WARNING: GitHub status POST failed after retries (context=${context}, state=${state}, code=${code})"
        }
    }
}

def withStageStatus(Closure body) {
    def stageName = env.STAGE_NAME
    setGithubStatus(stageName, 'pending', 'In progress')
    try {
        body()
    } catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e) {
        setGithubStatus(stageName, 'error', 'Job cancelled or aborted')
        throw e
    } catch (Exception e) {
        setGithubStatus(stageName, 'failure', 'Stage failed')
        throw e
    }
}

def getDockerImageWithStatus(Map conf=[:]) {
    def stageName = env.STAGE_NAME ?: "Docker Image"
    setGithubStatus(stageName, 'pending', 'In progress')
    try {
        def result = getDockerImage(conf)
        setGithubStatus(stageName, 'success', 'Completed successfully')
        return result
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e) {
        echo "The job was cancelled or aborted"
        setGithubStatus(stageName, 'error', 'Job cancelled or aborted')
        throw e
    }
    catch (Exception ex) {
        echo "Error in getDockerImageWithStatus: ${ex.message}"
        setGithubStatus(stageName, 'failure', 'Stage failed')
        throw ex
    }
}

def runShell(String command){
    def responseCode = sh returnStatus: true, script: "${command} > tmp.txt"
    def output = readFile(file: "tmp.txt")
    return (output != "")
}

def buildHipClangJob(Map conf=[:]){
        /*
            The following is a workaround for git submodule updating for the fin module.  After Jenkins upgrade,
            many plugins started misbehaving, and submodules wouldn't get pulled.  This ensures that we always pull
            the fin submodule and fail silently when the submodule directory already has artifacts in it.
        */
        sh(script: "git submodule update --init --recursive || true")
        env.HSA_ENABLE_SDMA=0
        env.DOCKER_BUILDKIT=1
        def image
        def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
        if (conf.get("enforce_xnack_on", false)) {
            dockerOpts = dockerOpts + " --env HSA_XNACK=1"
        }
        def video_id = sh(returnStdout: true, script: 'getent group video | cut -d: -f3')
        def render_id = sh(returnStdout: true, script: 'getent group render | cut -d: -f3')
        dockerOpts = dockerOpts + " --group-add=${video_id} --group-add=${render_id} "
        echo "Docker flags: ${dockerOpts}"

        def variant = env.STAGE_NAME

        def needs_gpu = conf.get("needs_gpu", true)
        def dvc_pull = conf.get("dvc_pull", false)
        def build_timeout = conf.get("build_timeout", 420)

        def retimage
        setGithubStatus(variant, 'pending', 'In progress')
        try {
            try {
                (retimage, image) = getDockerImage(conf)
                if (needs_gpu) {
                    withDockerContainer(image: image, args: dockerOpts) {
                        timeout(time: 2, unit: 'MINUTES'){
                            sh 'rocminfo | tee rocminfo.log'
                            if ( !runShell('grep -n "gfx" rocminfo.log') ){
                                throw new Exception ("GPU not found")
                            }
                            else{
                                echo "GPU is OK"
                            }
                        }
                    }
                }
            }
            catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
                echo "The job was cancelled or aborted"
                setGithubStatus(variant, 'error', 'Job cancelled or aborted')
                throw e
            }
            catch(Exception ex) {
                (retimage, image) = getDockerImage(conf)
                if (needs_gpu) {
                    withDockerContainer(image: image, args: dockerOpts) {
                        timeout(time: 2, unit: 'MINUTES'){
                            sh 'rocminfo | tee rocminfo.log'
                            if ( !runShell('grep -n "gfx" rocminfo.log') ){
                                throw new Exception ("GPU not found")
                            }
                            else{
                                echo "GPU is OK"
                            }
                        }
                    }
                }
            }

            //grab root of node workspace. not guaranteed to be /var/jenkins
            String remote_root = env.WORKSPACE.substring(0, env.WORKSPACE.lastIndexOf("workspace/"))
            withDockerContainer(image: image, args: dockerOpts + " -v=${remote_root}:${remote_root}") {
                timeout(time: build_timeout, unit:'MINUTES')
                {
                    // We set LOGNAME here because under the hood dvc calls Python's getpass.getuser() object to
                    // create a unique hash to store its local cache in. When Jenkins runs this Docker container, it
                    // runs as a UID that doesn't have an entry in /etc/passwd within the container. getuser() throws
                    // an exception if it can't get the user name for the current user's UID, but it will check the
                    // LOGNAME environment variable and use that value if it's available.
                    // https://github.com/iterative/dvc/blob/3915fa26aa7d95d5cbe345e62846bfd82dccbfc7/dvc/repo/__init__.py#L646
                    // https://docs.python.org/3/library/getpass.html#getpass.getuser
                    if (dvc_pull) {
                        sh """
                            cd ${env.WORKSPACE}/${env.MIOPEN_DIR}
                            LOGNAME=temp-user dvc pull -v
                           """.stripIndent()
                    }
                    cmake_build(conf)
                }
            }
            setGithubStatus(variant, 'success', 'Completed successfully')
        }
        catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e) {
            throw e  // already set status above
        }
        catch (Exception ex) {
            setGithubStatus(variant, 'failure', 'Stage failed')
            throw ex
        }
        return retimage
}

def RunPerfTest(Map conf=[:]){
    def dockerOpts="--device=/dev/kfd --device=/dev/dri --group-add video --group-add render --cap-add=SYS_PTRACE --security-opt seccomp=unconfined"
    try {
        def docker_image = conf.get("docker_image")
        def miopen_install_path = conf.get("miopen_install_path", "/opt/rocm")
        def results_dir = conf.get("results_dir", "${env.WORKSPACE}/${env.MIOPEN_DIR}/results")
        withDockerRegistry([ credentialsId: "docker_test_cred", url: "" ]) {
            docker_image.pull()
        }
        echo "docker image: ${docker_image}"
        //grab root of node workspace. not guaranteed to be /var/jenkins
        String remote_root = env.WORKSPACE.substring(0, env.WORKSPACE.lastIndexOf("workspace/"))
        docker_image.inside(dockerOpts + " -v=${remote_root}:${remote_root}")
        {
            timeout(time: 100, unit: 'MINUTES')
            {
                ld_lib="${miopen_install_path}/lib"
                def filename = conf.get("filename")
                assert(filename.trim())
                def cmd = "export LD_LIBRARY_PATH=${ld_lib} && ${miopen_install_path}/share/miopen/bin/test_perf.py  --filename ${filename} --install_path ${miopen_install_path} --results_path ${results_dir}/perf_results"
                if(params.PERF_TEST_OVERRIDE != '')
                {
                    echo "Appending MIOpenDriver cmd env vars: ${params.PERF_TEST_OVERRIDE}"
                    cmd += " --override ${params.PERF_TEST_OVERRIDE}"
                }
                sh cmd
                archiveArtifacts artifacts: "results/perf_results/${filename}", allowEmptyArchive: true, fingerprint: true
                jenkins_url = "${env.artifact_path}/${env.JOB_BASE_NAME}/lastSuccessfulBuild/artifact/results/perf_results"
                if(params.COMPARE_TO_BASE)
                {
                  try {
                      sh "rm -rf ${results_dir}/old_results/"
                      sh "wget -P ${results_dir}/old_results/ ${jenkins_url}/${filename}"
                  }
                  catch (Exception err){
                      currentBuild.result = 'SUCCESS'
                  }

                  try{
                     sh "${miopen_install_path}/share/miopen/bin/test_perf.py --compare_results --old_results_path ${results_dir}/old_results --results_path ${results_dir}/perf_results --filename ${filename}"
                  }
                  catch (Exception err){
                      currentBuild.result = 'SUCCESS'
                  }
                  cleanWs()
                }
            }
        }
    }
    catch (org.jenkinsci.plugins.workflow.steps.FlowInterruptedException e){
        echo "The job was cancelled or aborted"
        throw e
    }
}

def sendTeamsFailureNotification(Map conf=[:]) {
    def teamsMessage = null
    def teamsColor = null
    def teamsFacts = null

    if (conf.get("buildTheRock", false)) {
        teamsMessage = 'TheRock Docker Promotion Failed'
        teamsColor = '#FF6600'
        teamsFacts = [
            [name: 'Build',           template: "#${env.BUILD_NUMBER}"],
            [name: 'TheRock hash',    template: "${env.THEROCK_FULL_HASH ?: 'unknown'}"],
            [name: 'CK hash',         template: "${env.CK_GIT_HASH ?: 'unknown'}"],
            [name: 'Duration',        template: "${currentBuild.durationString}"],
            [name: 'Previous result', template: "${currentBuild.previousBuild?.result ?: 'N/A'}"],
            [name: 'Build URL',       template: "${env.BUILD_URL}"]
        ]
    } else if (env.BRANCH_NAME == 'develop') {
        teamsMessage = 'MIOpen develop branch CI failed'
        teamsColor = '#FF0000'
        teamsFacts = [
            [name: 'Build',           template: "#${env.BUILD_NUMBER}"],
            [name: 'Commit',          template: "${env.GIT_COMMIT?.take(7) ?: 'unknown'}"],
            [name: 'TheRock hash',    template: "${env.THEROCK_PROMOTED_HASH ?: 'unknown'}"],
            [name: 'Duration',        template: "${currentBuild.durationString}"],
            [name: 'Previous result', template: "${currentBuild.previousBuild?.result ?: 'N/A'}"],
            [name: 'Build URL',       template: "${env.BUILD_URL}"]
        ]
    }

    if (teamsMessage) {
        withCredentials([string(credentialsId: 'TEAMS_WEBHOOK_URL', variable: 'TEAMS_WEBHOOK_URL')]) {
            office365ConnectorSend(
                webhookUrl: TEAMS_WEBHOOK_URL,
                message: teamsMessage,
                status: 'Failure',
                color: teamsColor,
                factDefinitions: teamsFacts
            )
        }
    }
}

// Selective-rerun helpers: on "Restart from Stage", reads the previous build's
// FlowNode graph to find stages that already passed. Uses FlowNode (not wfapi)
// because wfapi only exposes top-level stages, not parallel sub-stages.
// Returns empty set on first run, full success, or any error (fail-open).

// Detects failure via ErrorAction, getOutcome(), or ResultAction reflection.
@NonCPS
private def isStageMarkedFailed(def endNode) {
    if (endNode.getAction(ErrorAction)) return true

    try {
        def outcome = endNode.getOutcome()
        if (outcome != null) {
            def outcomeStr = outcome.toString()
            if (outcomeStr.contains('FAILURE') || outcomeStr.contains('ABORTED')) {
                return true
            }
        }
    } catch (Exception e) {
    }

    // Fallback: ResultAction via reflection
    try {
        def actions = endNode.getActions()
        def resultAction = actions.find { action ->
            action.getClass().getSimpleName() == 'ResultAction'
        }
        if (resultAction != null) {
            def result = resultAction.result?.toString()
            if (result && (result.contains('FAILURE') || result.contains('ABORTED'))) {
                return true
            }
        }
    } catch (Exception e) {
    }

    return false
}

// Returns stage names that completed successfully in the given build.
@NonCPS
def getPassedStagesFromBuild(def rawBuild) {
    def passed = [] as Set
    def execution = rawBuild?.execution
    if (!execution) return passed

    def startNodes = [:]
    def endNodes   = [:]
    def errorIds   = [] as Set

    def walker = new FlowGraphWalker(execution)
    def walkerIter = walker.iterator()
    while (walkerIter.hasNext()) {
        def flowNode = walkerIter.next()
        if (flowNode.getAction(ErrorAction)) {
            errorIds << flowNode.id
        }
        if (flowNode instanceof StepStartNode) {
            def label  = flowNode.getAction(LabelAction)
            def thread = flowNode.getAction(ThreadNameAction)
            if (label && !thread) {
                startNodes[flowNode.id] = label.displayName
            }
        } else if (flowNode instanceof StepEndNode) {
            endNodes[flowNode.startNode?.id] = flowNode
        }
    }

    for (def entry : startNodes.entrySet()) {
        def startId   = entry.key
        def stageName = entry.value
        def endNode   = endNodes[startId]
        if (!endNode) continue
        if (isStageMarkedFailed(endNode)) continue

        passed << stageName
    }
    return passed
}

// Consolidates passed stages across a chain of restarts (same commit).
@NonCPS
def getPassedStagesAcrossRestartChain(def startBuild) {
    def consolidatedPassed = [] as Set
    def visited = [] as Set
    def currentBuild = startBuild

    while (currentBuild != null && !visited.contains(currentBuild.number)) {
        visited << currentBuild.number

        def passedInThisBuild = getPassedStagesFromBuild(currentBuild)
        consolidatedPassed.addAll(passedInThisBuild)

        def restartCause = currentBuild?.getCauses()?.find { cause ->
            cause.getClass().getName().contains('RestartDeclarativePipeline')
        }
        if (restartCause) {
            currentBuild = restartCause.getOriginal()
        } else {
            break
        }
    }

    return consolidatedPassed
}

// Returns [passedStages: Set<String>, debugMsg: String]. Consolidates passed
// stages across restart chains. Empty set on non-restart or error (fail-open).
@NonCPS
def getPassedStagesFromPreviousBuild() {
    def passed = [] as Set
    def debugMsg = ""
    try {
        def restartCause = currentBuild.rawBuild?.getCauses()?.find { cause ->
            cause.getClass().getName().contains('RestartDeclarativePipeline')
        }
        if (!restartCause) {
            debugMsg = "not a restart build, running all stages"
            return [passedStages: passed, debugMsg: debugMsg]
        }

        def prevRun = restartCause.getOriginal()
        if (!prevRun) {
            debugMsg = "could not resolve restarted build, running all stages"
            return [passedStages: passed, debugMsg: debugMsg]
        }
        if (prevRun.result?.toString() == 'SUCCESS') {
            debugMsg = "previous build was SUCCESS, running all stages"
            return [passedStages: passed, debugMsg: debugMsg]
        }

        debugMsg = "restarting from build #${prevRun.number}"
        passed = getPassedStagesAcrossRestartChain(prevRun)
    } catch (Exception e) {
        debugMsg = "error (${e.message}), running all stages"
        return [passedStages: [] as Set, debugMsg: debugMsg]
    }
    return [passedStages: passed, debugMsg: debugMsg]
}

// Stage factory methods -- return Map<String,Closure> for parallel().
// Moved here to keep the Jenkinsfile under the 64 KB bytecode limit.
def addStageIf(Map stagesMap, boolean condition, String name, Closure body) {
    if (condition) stagesMap[name] = { stage(name) { body() } }
}

def packageAndStaticCheckStages(def pipelineParams, def pipelineEnv, def rocmnodeFn, def withWorkingDirFn) {
    def result = getPassedStagesFromPreviousBuild()
    def passedStages = result.passedStages
    echo "Selective rerun: ${result.debugMsg}"
    echo "Selective rerun: passedStages (${passedStages.size()}): ${passedStages}"
    def stages = [:]

    def hipPackage = 'HIP Package'
    addStageIf(stages, !passedStages.contains(hipPackage), hipPackage) {
        node(rocmnodeFn("nogpu")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(package_build: true, needs_gpu: false, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def hipNoGpuDebug = 'HipNoGPU Debug Build Test'
    addStageIf(stages, pipelineParams.TARGET_NOGPU && !passedStages.contains(hipNoGpuDebug), hipNoGpuDebug) {
        node(rocmnodeFn("nogpu")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        def hipNoGpuFlags = "-DMIOPEN_BACKEND=HIPNOGPU -DMIOPEN_INSTALL_CXX_HEADERS=On"
                        def buildCmd = "ninja -j\$(nproc)"
                        buildHipClangJob(build_type: 'debug', setup_flags: hipNoGpuFlags, build_cmd: buildCmd, needs_gpu: false, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def tunaFinBuild = 'Tuna Fin Build Test'
    addStageIf(stages, !passedStages.contains(tunaFinBuild), tunaFinBuild) {
        node(rocmnodeFn("nogpu")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: "-DMIOPEN_BACKEND=HIPNOGPU", make_targets: "all", build_fin: "ON", needs_gpu: false, build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32NockBuild = 'Fp32 Hip Debug NOCK Build-Only'
    addStageIf(stages, !passedStages.contains(fp32NockBuild), fp32NockBuild) {
        node(rocmnodeFn("nogpu")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(build_type: 'debug', setup_flags: "-DMIOPEN_USE_COMPOSABLEKERNEL=Off", make_targets: "", build_install: true, needs_gpu: false, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    return stages
}

def fullTestStages(def pipelineParams, def pipelineEnv, def rocmnodeFn, def withWorkingDirFn, def runDbSyncJobFn, def runBuildAndSingleGtestJobFn) {
    def result = getPassedStagesFromPreviousBuild()
    def passedStages = result.passedStages
    echo "Selective rerun: ${result.debugMsg}"
    echo "Selective rerun: passedStages (${passedStages.size()}): ${passedStages}"
    def stages = [:]

    def Full_test    = pipelineEnv.Full_test
    def Bf16_flags   = pipelineEnv.Bf16_flags
    def Fp16_flags   = pipelineEnv.Fp16_flags
    def Tf32_flags   = pipelineEnv.Tf32_flags
    def gfx90a_flags = pipelineEnv.gfx90a_flags
    def gfx942_flags = pipelineEnv.gfx942_flags
    def gfx1101_flags = pipelineEnv.gfx1101_flags
    def Build_timeout_minutes = pipelineEnv.Build_timeout_minutes as Integer

    def hipTidy = 'Hip Tidy'
    addStageIf(stages, pipelineParams.RUN_HIP_TIDY && !passedStages.contains(hipTidy), hipTidy) {
        node(rocmnodeFn("nogpu")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        def setupCmd = "CXX='/opt/rocm/llvm/bin/clang++' cmake -G Ninja -DCMAKE_PREFIX_PATH=/opt/rocm -DMIOPEN_BACKEND=HIP -DBUILD_DEV=On .. "
                        def buildCmd = "ninja -j\$(nproc) -k 0 analyze"
                        buildHipClangJob(setup_cmd: setupCmd, build_cmd: buildCmd, needs_gpu: false, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    // GFX90A Tests
    def dbsyncGfx90a = 'Dbsync gfx90a'
    addStageIf(stages, pipelineParams.DBSYNC_TEST && pipelineParams.TARGET_GFX90A && !passedStages.contains(dbsyncGfx90a), dbsyncGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus { runDbSyncJobFn(gfx90a_flags, "ci") }
            } finally { cleanWs() }
        }
    }

    def bf16Gfx90a = 'Bf16 Hip Install All gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A && pipelineParams.DATATYPE_BF16 && !passedStages.contains(bf16Gfx90a), bf16Gfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Bf16_flags + gfx90a_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp16Gfx90a = 'Fp16 Hip Install All gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A && pipelineParams.DATATYPE_FP16 && !passedStages.contains(fp16Gfx90a), fp16Gfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Fp16_flags + gfx90a_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp32Gfx90a = 'Fp32 Hip Install All gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A && pipelineParams.DATATYPE_FP32 && !passedStages.contains(fp32Gfx90a), fp32Gfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + gfx90a_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    // GFX942 Tests
    def dbsyncGfx942 = 'Dbsync gfx942'
    addStageIf(stages, pipelineParams.DBSYNC_TEST && pipelineParams.TARGET_GFX942 && !passedStages.contains(dbsyncGfx942), dbsyncGfx942) {
        node(rocmnodeFn("gfx942")) {
            try {
                withStageStatus { runDbSyncJobFn(gfx942_flags, "ci") }
            } finally { cleanWs() }
        }
    }

    def bf16Gfx942 = 'Bf16 Hip Install All gfx942'
    addStageIf(stages, pipelineParams.TARGET_GFX942 && pipelineParams.DATATYPE_BF16 && !passedStages.contains(bf16Gfx942), bf16Gfx942) {
        node(rocmnodeFn("gfx942")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Bf16_flags + gfx942_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp16Gfx942 = 'Fp16 Hip Install All gfx942'
    addStageIf(stages, pipelineParams.TARGET_GFX942 && pipelineParams.DATATYPE_FP16 && !passedStages.contains(fp16Gfx942), fp16Gfx942) {
        node(rocmnodeFn("gfx942")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Fp16_flags + gfx942_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp32Gfx942 = 'Fp32 Hip Install All gfx942'
    addStageIf(stages, pipelineParams.TARGET_GFX942 && pipelineParams.DATATYPE_FP32 && !passedStages.contains(fp32Gfx942), fp32Gfx942) {
        node(rocmnodeFn("gfx942")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + gfx942_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def tf32Gfx942 = 'TF32 Hip Install All gfx942'
    addStageIf(stages, pipelineParams.TARGET_GFX942 && pipelineParams.DATATYPE_TF32 && !passedStages.contains(tf32Gfx942), tf32Gfx942) {
        node(rocmnodeFn("gfx942")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Tf32_flags + gfx942_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    // GFX1101 Tests
    def fp16Gfx1101 = 'Fp16 Hip Install All gfx1101'
    addStageIf(stages, pipelineParams.TARGET_NAVI32 && pipelineParams.DATATYPE_FP16 && !passedStages.contains(fp16Gfx1101), fp16Gfx1101) {
        node(rocmnodeFn("navi32")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Fp16_flags + gfx1101_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp32Gfx1101 = 'Fp32 Hip Install All gfx1101'
    addStageIf(stages, pipelineParams.TARGET_NAVI32 && pipelineParams.DATATYPE_FP32 && !passedStages.contains(fp32Gfx1101), fp32Gfx1101) {
        node(rocmnodeFn("navi32")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + gfx1101_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    return stages
}

def nightlyTestStages(def pipelineParams, def pipelineEnv, def rocmnodeFn, def withWorkingDirFn) {
    def stages = [:]

    def gfx90a_flags  = pipelineEnv.gfx90a_flags
    def gfx942_flags  = pipelineEnv.gfx942_flags
    def NOMLIR_flags  = pipelineEnv.NOMLIR_flags
    def Smoke_targets = pipelineEnv.Smoke_targets

    addStageIf(stages, true, 'Mark Build As Nightly') {
        node(rocmnodeFn("nogpu")) {
            try {
                withWorkingDirFn { currentBuild.description = "Nightly Build" }
            } finally { cleanWs() }
        }
    }

    def fp32NomlirGfx90a = 'Fp32 Hip Debug NOMLIR gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A, fp32NomlirGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        def nomlirBuildCmd = "CTEST_PARALLEL_LEVEL=4 MIOPEN_LOG_LEVEL=5 ninja -j\$(nproc) check"
                        buildHipClangJob(build_type: 'debug', setup_flags: NOMLIR_flags + gfx90a_flags, build_cmd: nomlirBuildCmd, test_flags: ' --verbose ', build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32StaticGfx90a = 'Fp32 Hip Static gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A, fp32StaticGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: "-DBUILD_SHARED_LIBS=Off" + gfx90a_flags, mlir_build: 'OFF', build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32NormalFindGfx90a = 'Fp32 Hip Normal-Find gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A, fp32NormalFindGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: gfx90a_flags, make_targets: "test_conv2d", execute_cmd: "bin/test_conv2d --disable-verification-cache", find_mode: "Normal", build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32FastFindGfx90a = 'Fp32 Hip Fast-Find gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A, fp32FastFindGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: gfx90a_flags, make_targets: "test_conv2d", execute_cmd: "MIOPEN_FIND_MODE=2 CTEST_PARALLEL_LEVEL=4 bin/test_conv2d --disable-verification-cache", build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32SqlitePerfdbGfx90a = 'Fp32 Hip SqlitePerfdb gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A, fp32SqlitePerfdbGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(make_targets: Smoke_targets, setup_flags: "-DMIOPEN_USE_SQLITE_PERF_DB=On" + gfx90a_flags, build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32FinInterfaceGfx90a = 'Fp32 Hip Fin Interface gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A, fp32FinInterfaceGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: "-DMIOPEN_ENABLE_FIN_INTERFACE=On" + gfx90a_flags, make_targets: "test_unit_FinInterface", execute_cmd: "bin/test_unit_FinInterface", gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32DebugGfx90a = 'Fp32 Hip Debug gfx90a'
    addStageIf(stages, pipelineParams.TARGET_GFX90A, fp32DebugGfx90a) {
        node(rocmnodeFn("gfx90a")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: gfx90a_flags, build_type: 'debug', make_targets: Smoke_targets, build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    def fp32DebugGfx942 = 'Fp32 Hip Debug gfx942'
    addStageIf(stages, pipelineParams.TARGET_GFX942, fp32DebugGfx942) {
        node(rocmnodeFn("gfx942")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: gfx942_flags, build_type: 'debug', make_targets: Smoke_targets, build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    return stages
}

def nonCriticalHWNightlyStages(def pipelineParams, def pipelineEnv, def rocmnodeFn, def withWorkingDirFn, def runDbSyncJobFn, def runBuildAndSingleGtestJobFn) {
    def result = getPassedStagesFromPreviousBuild()
    def passedStages = result.passedStages
    echo "Selective rerun: ${result.debugMsg}"
    echo "Selective rerun: passedStages (${passedStages.size()}): ${passedStages}"
    def stages = [:]

    def Full_test       = pipelineEnv.Full_test
    def Bf16_flags      = pipelineEnv.Bf16_flags
    def Fp16_flags      = pipelineEnv.Fp16_flags
    def gfx908_flags    = pipelineEnv.gfx908_flags
    def gfx1151_flags   = pipelineEnv.gfx1151_flags
    def Smoke_targets   = pipelineEnv.Smoke_targets
    def Build_timeout_minutes = pipelineEnv.Build_timeout_minutes as Integer

    addStageIf(stages, true, 'Mark Build As Nightly') {
        node(rocmnodeFn("nogpu")) {
            try {
                withWorkingDirFn { currentBuild.description = "Non-Critical HW Nightly Build" }
            } finally { cleanWs() }
        }
    }

    // GFX908 Tests
    def dbsyncGfx908 = 'Dbsync gfx908'
    addStageIf(stages, pipelineParams.DBSYNC_TEST && pipelineParams.TARGET_GFX908 && !passedStages.contains(dbsyncGfx908), dbsyncGfx908) {
        node(rocmnodeFn("gfx908")) {
            try {
                withStageStatus { runDbSyncJobFn(gfx908_flags, "ci") }
            } finally { cleanWs() }
        }
    }

    def bf16Gfx908 = 'Bf16 Hip Install All gfx908'
    addStageIf(stages, pipelineParams.TARGET_GFX908 && pipelineParams.DATATYPE_BF16 && !passedStages.contains(bf16Gfx908), bf16Gfx908) {
        node(rocmnodeFn("gfx908")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Bf16_flags + gfx908_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp16Gfx908 = 'Fp16 Hip Install All gfx908'
    addStageIf(stages, pipelineParams.TARGET_GFX908 && pipelineParams.DATATYPE_FP16 && !passedStages.contains(fp16Gfx908), fp16Gfx908) {
        node(rocmnodeFn("gfx908")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + Fp16_flags + gfx908_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp32Gfx908 = 'Fp32 Hip Install All gfx908'
    addStageIf(stages, pipelineParams.TARGET_GFX908 && pipelineParams.DATATYPE_FP32 && !passedStages.contains(fp32Gfx908), fp32Gfx908) {
        node(rocmnodeFn("gfx908")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: Full_test + gfx908_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp32DebugGfx908 = 'Fp32 Hip Debug gfx908'
    addStageIf(stages, pipelineParams.TARGET_GFX908 && !passedStages.contains(fp32DebugGfx908), fp32DebugGfx908) {
        node(rocmnodeFn("gfx908")) {
            try {
                withStageStatus {
                    withWorkingDirFn {
                        buildHipClangJob(setup_flags: gfx908_flags, build_type: 'debug', make_targets: Smoke_targets, build_install: true, gpu_family: "ci")
                    }
                }
            } finally { cleanWs() }
        }
    }

    // GFX115X Strix Halo Tests
    def bf16Gfx115X = 'Bf16 Hip Install All gfx115X'
    addStageIf(stages, pipelineParams.TARGET_NAVI35 && pipelineParams.DATATYPE_BF16 && !passedStages.contains(bf16Gfx115X), bf16Gfx115X) {
        node(rocmnodeFn("strix")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: " -DMIOPEN_TEST_GFX115X=On " + Full_test + Bf16_flags + gfx1151_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp16Gfx115X = 'Fp16 Hip Install All gfx115X'
    addStageIf(stages, pipelineParams.TARGET_NAVI35 && pipelineParams.DATATYPE_FP16 && !passedStages.contains(fp16Gfx115X), fp16Gfx115X) {
        node(rocmnodeFn("strix")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: " -DMIOPEN_TEST_GFX115X=On " + Full_test + Fp16_flags + gfx1151_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    def fp32Gfx115X = 'Fp32 Hip Install All gfx115X'
    addStageIf(stages, pipelineParams.TARGET_NAVI35 && pipelineParams.DATATYPE_FP32 && !passedStages.contains(fp32Gfx115X), fp32Gfx115X) {
        node(rocmnodeFn("strix")) {
            try {
                withStageStatus { runBuildAndSingleGtestJobFn(flags: " -DMIOPEN_TEST_GFX115X=On " + Full_test + gfx1151_flags, build_timeout_minutes: Build_timeout_minutes, gpu_family: "ci") }
            } finally { cleanWs() }
        }
    }

    return stages
}

return this
