/*******************************************************************************
 *
 * Copyright (C) 2020-2022 Advanced Micro Devices, Inc. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
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

// This shared library is available at https://github.com/ROCmSoftwarePlatform/rocJENKINS/
@Library('rocJenkins@pong') _

// This is file for internal AMD use.
// If you are interested in running your own Jenkins, please raise a github issue for assistance.

import com.amd.project.*
import com.amd.docker.*
import java.nio.file.Path

def runCompileCommand(platform, project, jobName, boolean debug=false) {
    project.paths.construct_build_prefix()

    command = """#!/usr/bin/env bash
            set -ex
            hostname
            cd ${project.paths.project_build_prefix}

            export HOME=/home/jenkins

            gfx_name="gfx90a"
            logic_path="library/src/blas3/Tensile/Logic/asm_full"
            repo_name="rocBLAS"

            git clone --depth=1 https://github.com/ROCm/\$repo_name.git ../\$repo_name

            TENSILE_PROFILE=ON Tensile/bin/TensileCreateLibrary \
              \$PWD/../\$repo_name/\$logic_path \
              _build \
              HIP \
              --merge-files \
              --separate-architecture \
              --lazy-library-loading \
              --no-short-file-names \
              --code-object-version=default \
              --cxx-compiler=amdclang++ \
              --jobs=32 \
              --library-format=msgpack \
              --architecture=\$gfx_name
            """

    platform.runCommand(this, command)

    archiveArtifacts artifacts: '**/*.prof'
}

def runCI(nodeDetails, jobName) {
    def prj = new rocProject('Tensile', 'Profiling')

    // Define test architectures; an optional ROCm version argument is available
    def nodes = new dockerNodes(nodeDetails, jobName, prj)

    boolean formatCheck = false
    boolean staticAnalysis = false

    prj.timeout.test = 30
    prj.defaults.ccache = false

    def compileCommand = { platform, project -> runCompileCommand(platform, project, jobName, false) }

    buildProject(prj, formatCheck, nodes.dockerArray, compileCommand, null, null, staticAnalysis)
}

def ci() {
    String urlJobName = auxiliary.getTopJobName(env.BUILD_URL)

    properties(auxiliary.addCommonProperties([]))
    runCI([ubuntu22:['gfx90a']], urlJobName)
}
