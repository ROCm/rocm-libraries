#include <flatbuffers/flatbuffers.h>
#include <hipdnn_flatbuffers_sdk/data_objects/uhd_generated.h>
#include <iostream>
#include <vector>

using namespace hipdnn_flatbuffers_sdk::data_objects;

int main() {
    flatbuffers::FlatBufferBuilder builder(2048);

    auto id = builder.CreateString("test_id");
    auto name = builder.CreateString("Test");
    auto hash = builder.CreateString("sha256:test");
    auto obj = builder.CreateString("max");
    auto units = builder.CreateString("tflops");
    auto xform = builder.CreateString("log1p");

    std::vector<flatbuffers::Offset<flatbuffers::String>> sigVec;
    sigVec.push_back(builder.CreateString("$kernel.tile_m"));
    auto featSig = builder.CreateVector(sigVec);

    std::vector<flatbuffers::Offset<UhdDerivedEntry>> derivedVec;
    auto derivedName = builder.CreateString("num_tiles");
    // RFC 0019 §6.1: Dims are positional ($q.dims[i] instead of named dimensions)
    auto derivedExpr = builder.CreateString(R"({"ceil_div": ["$q.dims[2]", "$kernel.tile_m"]})");
    derivedVec.push_back(CreateUhdDerivedEntry(builder, derivedName, derivedExpr));
    auto derivedOffset = builder.CreateVector(derivedVec);

    auto scoreOffset = CreateUhdScoreMetadata(builder, units, true, xform);

    auto uhdOffset = CreateUHD(builder, id, name, UhdAdapter::TREE_DATA, derivedOffset, featSig, hash, obj, scoreOffset);
    builder.Finish(uhdOffset, "HUHD");

    uint8_t* buf = builder.GetBufferPointer();
    size_t size = builder.GetSize();

    flatbuffers::Verifier verifier(buf, size);
    bool ok = verifier.VerifyBuffer<UHD>(nullptr);

    std::cout << "Buffer size: " << size << std::endl;
    std::cout << "Verification: " << (ok ? "PASS" : "FAIL") << std::endl;

    if(ok) {
        auto uhd = flatbuffers::GetRoot<UHD>(buf);
        std::cout << "ID: " << (uhd->id() ? uhd->id()->c_str() : "NULL") << std::endl;
        std::cout << "Derived count: " << (uhd->derived() ? uhd->derived()->size() : 0) << std::endl;

        // Now test UhdLoader::loadFromBuffer just like the test does
        std::vector<uint8_t> bufVec(buf, buf + size);
        std::cout << "\nCalling UhdLoader::loadFromBuffer..." << std::endl;
        // Can't actually call it here without linking the backend library
        // But we can at least check the buffer is valid
    }

    return ok ? 0 : 1;
}
