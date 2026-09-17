# Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
# SPDX-License-Identifier: MIT

"""Unit tests for codegen/sources/ -- the EngineSpec-producing adapter
interface (Task 2A.3): InteractiveAdapter (no inference) and HiprtcAdapter
(scans a .cpp for __global__ entry points and candidate KMD fields)."""

from codegen.sources import HiprtcAdapter, InteractiveAdapter, SourceAdapterResult


class TestInteractiveAdapter:
    def test_infer_returns_empty_result_regardless_of_input(self, tmp_path):
        adapter = InteractiveAdapter()
        result = adapter.infer(tmp_path / "anything.cpp")
        assert result == SourceAdapterResult(kernels=[], suggested_pack_count=1)

    def test_infer_with_no_sources_returns_empty_result(self):
        adapter = InteractiveAdapter()
        result = adapter.infer()
        assert result.kernels == []


class TestHiprtcAdapter:
    def _write(self, tmp_path, name, text):
        path = tmp_path / name
        path.write_text(text)
        return path

    def test_finds_single_entry_point(self, tmp_path):
        source = self._write(
            tmp_path,
            "ScaleAdd.cpp",
            'extern "C" __global__ void ScaleAdd(const float* x, float* y) {}\n',
        )
        result = HiprtcAdapter().infer(source)
        assert len(result.kernels) == 1
        assert result.kernels[0].entry_point == "ScaleAdd"
        assert result.kernels[0].source_file == "ScaleAdd.cpp"

    def test_finds_multiple_entry_points_in_one_file(self, tmp_path):
        source = self._write(
            tmp_path,
            "Multi.cpp",
            'extern "C" __global__ void KernelA(float* y) {}\n'
            'extern "C" __global__ void KernelB(float* y) {}\n',
        )
        result = HiprtcAdapter().infer(source)
        names = {k.entry_point for k in result.kernels}
        assert names == {"KernelA", "KernelB"}

    def test_externally_supplied_define_becomes_candidate_field(self, tmp_path):
        source = self._write(
            tmp_path,
            "ConvFwd.cpp",
            "// HIP_PLUGIN_CONV_TYPE and HIP_PLUGIN_CONV_BLOCK_SIZE come from the "
            "compile command.\n"
            'extern "C" __global__ void ConvFwd(const HIP_PLUGIN_CONV_TYPE* x) '
            "{\n  int block = HIP_PLUGIN_CONV_BLOCK_SIZE;\n}\n",
        )
        result = HiprtcAdapter().infer(source)
        assert "HIP_PLUGIN_CONV_TYPE" in result.kernels[0].template_params
        assert "HIP_PLUGIN_CONV_BLOCK_SIZE" in result.kernels[0].template_params

    def test_locally_defined_macro_is_not_a_candidate_field(self, tmp_path):
        """A #define this file sets itself is not externally supplied -- it is
        not a config-facing KMD field candidate."""
        source = self._write(
            tmp_path,
            "Local.cpp",
            "#define HIP_PLUGIN_LOCAL_CONST 42\n"
            'extern "C" __global__ void Local(float* y) { y[0] = HIP_PLUGIN_LOCAL_CONST; }\n',
        )
        result = HiprtcAdapter().infer(source)
        assert "HIP_PLUGIN_LOCAL_CONST" not in result.kernels[0].template_params

    def test_template_parameters_become_candidate_fields(self, tmp_path):
        source = self._write(
            tmp_path,
            "Templated.cpp",
            "template <int BlockSize, typename T>\n"
            'extern "C" __global__ void Templated(T* y) {}\n',
        )
        result = HiprtcAdapter().infer(source)
        assert "BlockSize" in result.kernels[0].template_params

    def test_no_entry_points_yields_empty_kernel_list(self, tmp_path):
        source = self._write(tmp_path, "NotAKernel.cpp", "void ordinaryFunction() {}\n")
        result = HiprtcAdapter().infer(source)
        assert result.kernels == []
        assert result.suggested_pack_count == 1

    def test_suggested_pack_count_is_distinct_source_file_count(self, tmp_path):
        a = self._write(
            tmp_path, "A.cpp", 'extern "C" __global__ void A(float* y) {}\n'
        )
        b = self._write(
            tmp_path, "B.cpp", 'extern "C" __global__ void B(float* y) {}\n'
        )
        result = HiprtcAdapter().infer(a, b)
        assert result.suggested_pack_count == 2

    def test_multiple_entry_points_one_file_is_one_pack(self, tmp_path):
        source = self._write(
            tmp_path,
            "OnePack.cpp",
            'extern "C" __global__ void Variant1(float* y) {}\n'
            'extern "C" __global__ void Variant2(float* y) {}\n',
        )
        result = HiprtcAdapter().infer(source)
        assert result.suggested_pack_count == 1
