// pyplugin.cpp - shared library that embeds Python via pybind11
#include <pybind11/embed.h>
#include <pybind11/pybind11.h>

#include <cstdio>
#include <string>

namespace py = pybind11;

// Hold interpreter for process lifetime (no Py_Finalize per plan risk register).
static py::scoped_interpreter* g_interp = nullptr;

extern "C" int run_python_smoke(void) {
    try {
        if (!Py_IsInitialized()) {
            g_interp = new py::scoped_interpreter{};
        }
        py::gil_scoped_acquire gil;

        // sys.version_info access
        py::module_ sys = py::module_::import("sys");
        py::object vi = sys.attr("version_info");
        int major = vi.attr("major").cast<int>();
        int minor = vi.attr("minor").cast<int>();
        std::printf("[pyplugin] Python %d.%d\n", major, minor);

        // json roundtrip
        py::module_ json = py::module_::import("json");
        py::dict d;
        d["ok"] = 1;
        std::string s = json.attr("dumps")(d).cast<std::string>();
        std::printf("[pyplugin] json.dumps -> %s\n", s.c_str());
        py::object parsed = json.attr("loads")(s);
        int ok = parsed["ok"].cast<int>();
        if (ok != 1) {
            std::fprintf(stderr, "[pyplugin] unexpected parsed value: %d\n", ok);
            return 2;
        }
        return 0;
    } catch (py::error_already_set& e) {
        std::fprintf(stderr, "[pyplugin] python error: %s\n", e.what());
        return 3;
    } catch (std::exception& e) {
        std::fprintf(stderr, "[pyplugin] std error: %s\n", e.what());
        return 4;
    } catch (...) {
        std::fprintf(stderr, "[pyplugin] unknown error\n");
        return 5;
    }
}
