// Copyright Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

// ContractionProblemGemm declares `enum CONST`, a name the Windows SDK headers
// define as a macro, and msgpack.hpp reaches those headers. Every header that
// declares the enum therefore has to be seen before <msgpack.hpp>, which is why
// this block sits above the one holding SolutionState.hpp. clang-format sorts
// within an include block and preserves the order of the blocks themselves.
#include <Tensile/AMDGPUPredicates.hpp>
#include <Tensile/ContractionProblemPredicates.hpp>
#include <Tensile/ContractionSolution.hpp>
#include <Tensile/ContractionTaskPredicates.hpp>
#include <Tensile/KernelLanguageTypes.hpp>

#include <Tensile/msgpack/SolutionState.hpp>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace TensileLite
{
    namespace
    {
        using msgpack::type::object_type;

        using ProblemPredicate  = Predicates::Predicate<ContractionProblemGemm>;
        using TaskPredicate     = Predicates::Predicate<Task>;
        using HardwarePredicate = Predicates::Predicate<Hardware>;
        using AMDGPUPredicate   = Predicates::Predicate<AMDGPU>;

        // internalParameters["GlobalSplitUPGR"] in Tensile/Common/GlobalParameters.py.
        constexpr size_t GlobalSplitUPGR = 16;

        // SUPPORTED_BUILD_CHIP_IDS in Tensile/Common/Architectures.py.
        constexpr std::array<char const*, 14> SupportedBuildChipIds = {
            "74a0", "74a1", "74a2", "74a3", "74a5", "74a9", "75a0",
            "75b0", "75a2", "75b2", "75a3", "75b3", "75a8", "75b8"};

        // DataType.properties in Tensile/Common/DataType.py, by enum value: the
        // enum name and char it is looked up by, and whether numBytes() > 1.
        struct DataTypeProperties
        {
            char const* name;
            char const* abbrev;
            bool        wide;
        };
        constexpr DataTypeProperties DataTypes[] = {
            {"Float", "S", true},
            {"Double", "D", true},
            {"ComplexFloat", "C", true},
            {"ComplexDouble", "Z", true},
            {"Half", "H", true},
            {"Int8x4", "4xi8", true},
            {"Int32", "I", true},
            {"BFloat16", "B", true},
            {"Int8", "I8", false},
            {"Int64", "I64", true},
            {"XFloat32", "X", true},
            {"Float8_fnuz", "F8N", false},
            {"BFloat8_fnuz", "B8N", false},
            {"Float8BFloat8_fnuz", "F8B8N", false},
            {"BFloat8Float8_fnuz", "B8F8N", false},
            {"Float8", "F8", false},
            {"BFloat8", "B8", false},
            {"Float8BFloat8", "F8B8", false},
            {"BFloat8Float8", "B8F8", false},
            {"Float6", "F6", false},
            {"BFloat6", "B6", false},
            {"Float4", "F4", false},
            {"E8", "E8", false},
            {"E5M3", "E5M3", false},
        };
        static_assert(std::size(DataTypes) == static_cast<size_t>(rocisa::DataType::Count));

        std::string_view view(msgpack::object const& object)
        {
            return {object.via.str.ptr, object.via.str.size};
        }

        bool isFloat(msgpack::object const& object)
        {
            return object.type == object_type::FLOAT32 || object.type == object_type::FLOAT64;
        }

        // One value of the state and where it sits, read with Python's semantics.
        class Value
        {
        public:
            Value(msgpack::object const& object, std::string path)
                : m_object(&object)
                , m_path(std::move(path))
            {
            }

            msgpack::object const& object() const
            {
                return *m_object;
            }

            std::string const& path() const
            {
                return m_path;
            }

            [[noreturn]] void fail(std::string const& what) const
            {
                throw std::runtime_error("Solution state " + (m_path.empty() ? "" : m_path + " ")
                                         + "is not " + what);
            }

            bool isNone() const
            {
                return m_object->type == object_type::NIL
                       || (m_object->type == object_type::STR && view(*m_object) == "null");
            }

            bool isNumber() const
            {
                switch(m_object->type)
                {
                case object_type::BOOLEAN:
                case object_type::POSITIVE_INTEGER:
                case object_type::NEGATIVE_INTEGER:
                    return true;
                default:
                    return isFloat(*m_object);
                }
            }

            bool truthy() const
            {
                if(isNone())
                    return false;
                switch(m_object->type)
                {
                case object_type::BOOLEAN:
                    return m_object->via.boolean;
                case object_type::POSITIVE_INTEGER:
                    return m_object->via.u64 != 0;
                case object_type::NEGATIVE_INTEGER:
                    return true;
                case object_type::FLOAT32:
                case object_type::FLOAT64:
                    return m_object->via.f64 != 0;
                case object_type::STR:
                    return m_object->via.str.size != 0;
                case object_type::BIN:
                    return m_object->via.bin.size != 0;
                case object_type::ARRAY:
                    return m_object->via.array.size != 0;
                case object_type::MAP:
                    return m_object->via.map.size != 0;
                default:
                    return true;
                }
            }

            int64_t integer() const
            {
                switch(m_object->type)
                {
                case object_type::BOOLEAN:
                    return m_object->via.boolean ? 1 : 0;
                case object_type::POSITIVE_INTEGER:
                    if(m_object->via.u64 > uint64_t(std::numeric_limits<int64_t>::max()))
                        fail("a 64-bit integer");
                    return int64_t(m_object->via.u64);
                case object_type::NEGATIVE_INTEGER:
                    return m_object->via.i64;
                default:
                    fail("an integer");
                }
            }

            double number() const
            {
                if(!isNumber())
                    fail("a number");
                return isFloat(*m_object) ? m_object->via.f64 : double(integer());
            }

            std::string text() const
            {
                if(m_object->type == object_type::STR)
                    return std::string(view(*m_object));
                if(isFloat(*m_object) && m_object->via.f64 == 0)
                    return "";
                fail("a string");
            }

            bool isText(std::string_view text) const
            {
                return m_object->type == object_type::STR && view(*m_object) == text;
            }

            bool equals(double number) const
            {
                return isNumber() && this->number() == number;
            }

            // Python's value == other.
            bool equals(Value const& other) const
            {
                if(isNumber() && other.isNumber())
                    return number() == other.number();
                return *m_object == other.object();
            }

            std::vector<Value> list() const
            {
                if(m_object->type != object_type::ARRAY)
                    fail("a list");
                std::vector<Value> rv;
                rv.reserve(m_object->via.array.size);
                for(uint32_t i = 0; i < m_object->via.array.size; i++)
                    rv.emplace_back(m_object->via.array.ptr[i],
                                    m_path + "[" + std::to_string(i) + "]");
                return rv;
            }

            std::vector<int64_t> integers() const
            {
                std::vector<int64_t> rv;
                for(auto const& item : list())
                    rv.push_back(item.integer());
                return rv;
            }

            rocisa::DataType dataType() const
            {
                if(m_object->type == object_type::STR)
                {
                    std::string name = lower(text());
                    for(size_t i = 0; i < std::size(DataTypes); i++)
                        if(lower(DataTypes[i].name) == name || lower(DataTypes[i].abbrev) == name)
                            return static_cast<rocisa::DataType>(i);
                    fail("a data type");
                }
                auto value = integer();
                if(value < 0 || value >= static_cast<int64_t>(rocisa::DataType::Count))
                    fail("a data type");
                return static_cast<rocisa::DataType>(value);
            }

            static std::string lower(std::string text)
            {
                std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) {
                    return std::tolower(c);
                });
                return text;
            }

        private:
            msgpack::object const* m_object;
            std::string            m_path;
        };

        // A map of the state, keeping the stored order Python iterates it in.
        class Dict
        {
        public:
            explicit Dict(Value const& value)
                : m_path(value.path())
            {
                if(value.object().type != object_type::MAP)
                    value.fail("a map");
                auto const& map = value.object().via.map;
                for(uint32_t i = 0; i < map.size; i++)
                {
                    auto const& key = map.ptr[i].key;
                    if(key.type != object_type::STR)
                        value.fail("a map with string keys");
                    std::string name(view(key));
                    m_items.emplace_back(name, Value(map.ptr[i].val, child(name)));
                }
            }

            std::vector<std::pair<std::string, Value>> const& items() const
            {
                return m_items;
            }

            Value const* find(std::string_view key) const
            {
                for(auto const& item : m_items)
                    if(item.first == key)
                        return &item.second;
                return nullptr;
            }

            bool has(std::string_view key) const
            {
                return find(key) != nullptr;
            }

            Value const& operator[](std::string_view key) const
            {
                if(auto const* value = find(key))
                    return *value;
                throw std::runtime_error("Solution state has no " + child(std::string(key)));
            }

            Dict dict(std::string_view key) const
            {
                return Dict((*this)[key]);
            }

            int64_t integer(std::string_view key) const
            {
                return (*this)[key].integer();
            }

            int64_t integer(std::string_view key, int64_t fallback) const
            {
                auto const* value = find(key);
                return value ? value->integer() : fallback;
            }

            bool truthy(std::string_view key) const
            {
                return (*this)[key].truthy();
            }

            bool truthy(std::string_view key, bool fallback) const
            {
                auto const* value = find(key);
                return value ? value->truthy() : fallback;
            }

            std::string text(std::string_view key) const
            {
                return (*this)[key].text();
            }

            rocisa::DataType dataType(std::string_view key) const
            {
                return (*this)[key].dataType();
            }

        private:
            std::string child(std::string const& key) const
            {
                return m_path.empty() ? key : m_path + "." + key;
            }

            std::string                                m_path;
            std::vector<std::pair<std::string, Value>> m_items;
        };

        Dict emptyDict()
        {
            static msgpack::object const map = [] {
                msgpack::object rv;
                rv.type         = object_type::MAP;
                rv.via.map.size = 0;
                rv.via.map.ptr  = nullptr;
                return rv;
            }();
            return Dict(Value(map, ""));
        }

        // Python's a / b, which raises on a zero divisor.
        double quotient(Value const& a, Value const& b)
        {
            if(b.number() == 0)
                b.fail("a nonzero divisor");
            return a.number() / b.number();
        }

        // Python's a // b on integers.
        int64_t floorQuotient(Value const& a, Value const& b)
        {
            auto numerator   = a.integer();
            auto denominator = b.integer();
            if(denominator == 0)
                b.fail("a nonzero divisor");
            auto rv = numerator / denominator;
            if(numerator % denominator != 0 && ((numerator < 0) != (denominator < 0)))
                rv--;
            return rv;
        }

        template <typename Enum, typename Name>
        Enum enumByName(Value const& value, int count, Name name, char const* what)
        {
            auto text = value.text();
            for(int i = 0; i < count; i++)
                if(name(static_cast<Enum>(i)) == text)
                    return static_cast<Enum>(i);
            value.fail(what);
        }

        // ActivationType.lookup in Tensile/Activation.py, in order.
        struct ActivationEntry
        {
            char const* name;
            bool        isGradient;
            int         extraArgs;
            // Half, Single, Double, BFloat16, Int8, Int16, Int32.
            std::array<bool, 7> can;
            bool                tensile;
            bool                hipblaslt;
        };

        constexpr ActivationEntry Activations[] = {
            {"none", false, 0, {true, true, true, true, true, true, true}, true, true},
            {"abs", false, 0, {true, true, true, true, false, false, true}, true, false},
            {"clippedrelu", false, 2, {true, true, true, false, false, false, true}, true, false},
            {"gelu", false, 0, {true, true, false, false, false, false, false}, true, true},
            {"leakyrelu", false, 1, {true, true, true, false, false, false, true}, true, false},
            {"relu", false, 0, {true, true, true, false, false, false, true}, true, true},
            {"sigmoid", false, 0, {true, true, false, false, false, false, false}, true, true},
            {"tanh", false, 2, {true, true, false, false, false, false, false}, true, false},
            {"dgelu", true, 0, {false, true, false, false, false, false, false}, true, true},
            {"drelu", true, 0, {false, true, false, false, false, false, false}, true, true},
            {"geluscaling", false, 1, {true, true, false, false, false, false, false}, true, false},
            {"silu", false, 0, {true, true, false, false, false, false, false}, true, true},
            {"swish", false, 1, {true, true, false, false, false, false, false}, true, false},
            {"clamp", false, 2, {true, true, true, false, false, false, true}, true, true},
            {"hipblaslt_all", false, 0, {}, false, true},
            {"all", false, 0, {}, true, true},
        };

        // ActivationType.lookupVeri.
        constexpr char const* VerificationActivation = "exp";

        ActivationEntry const* findActivation(std::string const& name)
        {
            for(auto const& entry : Activations)
                if(name == entry.name)
                    return &entry;
            return nullptr;
        }

        // ActivationType(value): the lower-case name.
        std::string activationName(Value const& value)
        {
            if(value.object().type != object_type::STR)
                value.fail("an activation type");
            auto name = Value::lower(value.text());
            if(!findActivation(name) && name != VerificationActivation)
                value.fail("an activation type");
            return name;
        }

        bool isAllActivations(std::string const& name)
        {
            return name == "all" || name == "hipblaslt_all";
        }

        // ActivationType.getAdditionalArgNum() for Export.NORMAL.
        int additionalArgNum(std::string const& name)
        {
            if(isAllActivations(name))
            {
                int rv = 0;
                for(auto const& entry : Activations)
                    if(!entry.isGradient)
                        rv = std::max(rv, entry.extraArgs);
                return rv;
            }
            auto const* entry = findActivation(name);
            return entry ? entry->extraArgs : 0;
        }

        bool typeAvailable(ActivationEntry const& entry, rocisa::DataType type)
        {
            return (type == rocisa::DataType::Half && entry.can[0])
                   || (type == rocisa::DataType::Float && entry.can[1])
                   || (type == rocisa::DataType::Double && entry.can[2])
                   || (type == rocisa::DataType::BFloat16 && entry.can[3])
                   || (type == rocisa::DataType::Int8 && entry.can[4])
                   || (type == rocisa::DataType::Int32 && entry.can[6]);
        }

        // The C++ activation an ActivationType state names, by its capitalize().
        ActivationType activationType(std::string const& name)
        {
            std::string capitalized = name;
            if(!capitalized.empty())
                capitalized[0] = std::toupper(static_cast<unsigned char>(capitalized[0]));
            for(int i = 0; i < static_cast<int>(ActivationType::Count); i++)
                if(ToString(static_cast<ActivationType>(i)) == capitalized)
                    return static_cast<ActivationType>(i);
            throw std::runtime_error("No activation type is named " + capitalized);
        }

        // ActivationType.getEnumStrList(dataType, supportedBy, exportType=...).
        std::vector<ActivationType>
            activationEnumList(rocisa::DataType type, bool all, bool gradientOnly)
        {
            std::vector<ActivationType> rv;
            for(auto const& entry : Activations)
            {
                if(gradientOnly ? !entry.isGradient : entry.isGradient)
                    continue;
                if(isAllActivations(entry.name))
                    continue;
                bool supported = entry.hipblaslt || (all && entry.tensile);
                if(typeAvailable(entry, type) && supported)
                    rv.push_back(activationType(entry.name));
            }
            return rv;
        }

        bool wideType(rocisa::DataType type)
        {
            return DataTypes[static_cast<size_t>(type)].wide;
        }

        // list.sort() on a list of data types as the state spells them, then
        // DataType() on each.
        std::vector<rocisa::DataType> sortedDataTypes(Value const& value)
        {
            auto items = value.list();
            bool texts = !items.empty() && items.front().object().type == object_type::STR;
            for(auto const& item : items)
                if((item.object().type == object_type::STR) != texts)
                    value.fail("a list of data types of one kind");
            std::stable_sort(items.begin(), items.end(), [texts](Value const& a, Value const& b) {
                return texts ? a.text() < b.text() : a.number() < b.number();
            });
            std::vector<rocisa::DataType> rv;
            for(auto const& item : items)
                rv.push_back(item.dataType());
            return rv;
        }

        // sorted(set(types)).
        std::vector<rocisa::DataType> uniqueSorted(std::vector<rocisa::DataType> types)
        {
            std::sort(types.begin(), types.end());
            types.erase(std::unique(types.begin(), types.end()), types.end());
            return types;
        }

        // getBiasDataTypeListDefault in Tensile/SolutionStructs/Problem.py.
        std::vector<rocisa::DataType> biasDataTypeListDefault(Dict const& problem)
        {
            std::vector<rocisa::DataType> types;
            for(auto key : {"DataType", "ComputeDataType", "DestDataType"})
            {
                auto type = problem.dataType(key);
                if(wideType(type))
                    types.push_back(type);
            }
            return uniqueSorted(types);
        }

        // getGateResidualDataTypeListDefault in Tensile/SolutionStructs/Problem.py.
        std::vector<rocisa::DataType> gateResidualDataTypeListDefault(Dict const& problem)
        {
            return uniqueSorted(
                {problem.dataType("DataType"), problem.dataType("ComputeDataType")});
        }

        template <typename T>
        void assignSequence(Value const& value, T& target)
        {
            auto items = value.list();
            if(items.size() > 3)
                value.fail("a list of at most 3 values");
            using Element = decltype(target.x);
            if(items.size() > 0)
                target.x = static_cast<Element>(items[0].integer());
            if(items.size() > 1)
                target.y = static_cast<Element>(items[1].integer());
            if(items.size() > 2)
                target.z = static_cast<Element>(items[2].integer());
        }

        template <typename T, size_t N>
        void assignArray(Value const& value, std::array<T, N>& target)
        {
            auto items = value.list();
            if(items.size() > N)
                value.fail("a list of at most " + std::to_string(N) + " values");
            for(size_t i = 0; i < items.size(); i++)
                target[i] = static_cast<T>(items[i].integer());
        }

        template <typename P, typename V>
        std::shared_ptr<P> withValue(V value)
        {
            auto rv   = std::make_shared<P>();
            rv->value = std::move(value);
            return rv;
        }

        template <typename P, typename I, typename V>
        std::shared_ptr<P> withIndexValue(I index, V value)
        {
            auto rv   = std::make_shared<P>();
            rv->index = index;
            rv->value = std::move(value);
            return rv;
        }

        // Properties.Predicate.And.
        template <typename Object>
        std::shared_ptr<Predicates::Predicate<Object>>
            conjunction(std::vector<std::shared_ptr<Predicates::Predicate<Object>>> predicates)
        {
            if(predicates.empty())
                return std::make_shared<Predicates::True<Object>>();
            if(predicates.size() == 1)
                return predicates.front();
            return std::make_shared<Predicates::And<Object>>(predicates);
        }

        // Properties.Predicate.Or.
        template <typename Object>
        std::shared_ptr<Predicates::Predicate<Object>>
            disjunction(std::vector<std::shared_ptr<Predicates::Predicate<Object>>> predicates)
        {
            if(predicates.empty())
                return std::make_shared<Predicates::True<Object>>();
            if(predicates.size() == 1)
                return predicates.front();
            return std::make_shared<Predicates::Or<Object>>(predicates);
        }

        // Tensile/ExecutionPolicy.py.
        struct ExecutionPolicy
        {
            TileProcessingStrategy strategy   = TileProcessingStrategy::None;
            WorkAssignment         assignment = WorkAssignment::StaticGrid;

            bool persistent() const
            {
                return strategy != TileProcessingStrategy::None;
            }
            bool persistentDataParallel() const
            {
                return strategy == TileProcessingStrategy::DataParallel;
            }
            bool streamK() const
            {
                return strategy == TileProcessingStrategy::StreamK;
            }
        };

        // The selectors resolve_policy reads, by value.
        struct PolicySelectors
        {
            std::string  strategy   = "None";
            std::string  assignment = "StaticGrid";
            Value const* persistentLoop = nullptr;
        };

        ExecutionPolicy resolvePolicy(PolicySelectors const& selectors)
        {
            ExecutionPolicy policy;
            try
            {
                policy.strategy = parseTileProcessingStrategy(selectors.strategy);
            }
            catch(std::runtime_error const&)
            {
                throw std::runtime_error(
                    "TileProcessingStrategy must be 'None', DataParallel, or StreamK");
            }
            try
            {
                policy.assignment = parseWorkAssignment(selectors.assignment);
            }
            catch(std::runtime_error const&)
            {
                throw std::runtime_error(
                    "WorkAssignment must be StaticGrid, DynamicWorkQueue, or Hybrid");
            }
            if(policy.strategy == TileProcessingStrategy::None)
                policy.assignment = WorkAssignment::StaticGrid;
            else if(policy.persistentDataParallel()
                    && policy.assignment != WorkAssignment::StaticGrid)
                throw std::runtime_error(std::string(toString(policy.strategy))
                                         + " supports WorkAssignment=StaticGrid only");
            if(selectors.persistentLoop
               && !selectors.persistentLoop->equals(policy.persistent() ? 1 : 0))
                throw std::runtime_error(
                    "_PersistentLoop disagrees with the derived execution policy");
            return policy;
        }

        // A type(x) is int check: True and False are not ints there.
        bool isInt(Value const& value)
        {
            return value.object().type == object_type::POSITIVE_INTEGER
                   || value.object().type == object_type::NEGATIVE_INTEGER;
        }

        // What SizeMapping reads of normalize_execution_policy(d, regenerate=...).
        struct NormalizedPolicy
        {
            ExecutionPolicy policy;
            int64_t         streamKAtomic            = 0;
            int64_t         prefetchAcrossPersistent = 0;
        };

        NormalizedPolicy normalizeExecutionPolicy(Dict const& config, bool regenerate)
        {
            if(config.has("PersistentLoop"))
                throw std::runtime_error(
                    "PersistentLoop is derived from TileProcessingStrategy and WorkAssignment; "
                    "it is not a tuning parameter");
            bool legacy = config.has("StreamK") || config.has("StreamKForceDPOnly");

            PolicySelectors selectors;
            if(auto const* value = config.find("TileProcessingStrategy"))
                selectors.strategy = value->text();
            if(auto const* value = config.find("WorkAssignment"))
                selectors.assignment = value->text();
            selectors.persistentLoop = config.find("_PersistentLoop");

            if(legacy)
            {
                // _translate_legacy_streamk_selectors.
                auto const* modeValue  = config.find("StreamK");
                auto const* forceValue = config.find("StreamKForceDPOnly");
                int64_t     mode       = 0;
                int64_t     force      = 0;
                if(modeValue)
                {
                    if(!isInt(*modeValue))
                        throw std::runtime_error("Legacy StreamK must be 0, 3, 4, or 5; "
                                                 "modes 1 and 2 are retired");
                    mode = modeValue->integer();
                }
                if(mode != 0 && mode != 3 && mode != 4 && mode != 5)
                    throw std::runtime_error(
                        "Legacy StreamK must be 0, 3, 4, or 5; modes 1 and 2 are retired");
                if(forceValue)
                {
                    if(!isInt(*forceValue)
                       && forceValue->object().type != object_type::BOOLEAN)
                        throw std::runtime_error("StreamKForceDPOnly must be 0 or 1");
                    force = forceValue->integer();
                }
                if(force != 0 && force != 1)
                    throw std::runtime_error("StreamKForceDPOnly must be 0 or 1");
                if(force && (mode != 3 || config.truthy("StreamKAtomic", false)))
                    throw std::runtime_error("StreamKForceDPOnly requires non-atomic StreamK=3");
                std::string strategy = force ? "DataParallel" : mode == 0 ? "None" : "StreamK";
                std::string assignment = mode == 4   ? "DynamicWorkQueue"
                                         : mode == 5 ? "Hybrid"
                                                     : "StaticGrid";
                if(config.has("TileProcessingStrategy") && selectors.strategy != strategy)
                    throw std::runtime_error(
                        "Conflicting legacy StreamK and TileProcessingStrategy="
                        + selectors.strategy + " (expected " + strategy + ")");
                selectors.strategy = strategy;
                if(!(strategy == "None" && config.has("WorkAssignment")))
                {
                    if(config.has("WorkAssignment") && selectors.assignment != assignment)
                        throw std::runtime_error("Conflicting legacy StreamK and WorkAssignment="
                                                 + selectors.assignment + " (expected "
                                                 + assignment + ")");
                    selectors.assignment = assignment;
                }
            }

            // ALIASES: the old spelling wins, and must agree with the new one.
            Value const* workQueueStealing = config.find("WorkQueueStealing");
            for(auto alias : {std::pair<char const*, char const*>{"StreamKXCCMapping",
                                                                   "PersistentXCCMapping"},
                              std::pair<char const*, char const*>{"StreamKWorkStealing",
                                                                   "WorkQueueStealing"}})
            {
                auto const* old = config.find(alias.first);
                if(!old)
                    continue;
                auto const* current = config.find(alias.second);
                if(current && !old->equals(*current))
                    throw std::runtime_error(std::string("Conflicting ") + alias.first + " and "
                                             + alias.second);
                if(std::string_view(alias.second) == "WorkQueueStealing")
                    workQueueStealing = old;
            }

            NormalizedPolicy rv;
            rv.policy        = resolvePolicy(selectors);
            auto const& policy = rv.policy;

            for(auto option : {"StreamKAtomic", "StreamKFixupTreeReduction", "DebugStreamK"})
            {
                if(!policy.streamK() && !legacy && config.truthy(option, false))
                    throw std::runtime_error(std::string(option)
                                             + " requires TileProcessingStrategy=StreamK");
            }
            rv.streamKAtomic = policy.streamK() ? config.integer("StreamKAtomic", 0) : 0;

            if(workQueueStealing && workQueueStealing->truthy()
               && (!policy.streamK() || policy.assignment == WorkAssignment::StaticGrid))
            {
                if(!legacy || policy.persistent())
                    throw std::runtime_error(
                        "WorkQueueStealing requires StreamK with DynamicWorkQueue or Hybrid");
            }

            if(!policy.persistent())
            {
                for(auto option : {"PrefetchAcrossPersistent",
                                   "ReuseAcrossPersistent",
                                   "DebugPersistentKernelLoopForever"})
                {
                    if(!legacy && config.truthy(option, false))
                        throw std::runtime_error(std::string(option)
                                                 + " requires a persistent TileProcessingStrategy");
                }
            }
            rv.prefetchAcrossPersistent
                = policy.persistent() ? config.integer("PrefetchAcrossPersistent", 0) : 0;

            Dict support = config.has("InternalSupportParams")
                               ? config.dict("InternalSupportParams")
                               : emptyDict();
            int64_t version      = 0;
            int64_t outerVersion = 3;
            if(auto const* value = support.find("PersistentLoopArgsVersion"))
            {
                if(!isInt(*value))
                    throw std::runtime_error("Unsupported PersistentLoopArgsVersion");
                version = value->integer();
            }
            if(auto const* value = support.find("KernArgsVersion"))
            {
                if(!isInt(*value))
                    throw std::runtime_error("Unsupported KernArgsVersion");
                outerVersion = value->integer();
            }
            if(version != 0 && version != 1)
                throw std::runtime_error("Unsupported PersistentLoopArgsVersion");
            if(outerVersion < 0 || outerVersion > 3)
                throw std::runtime_error("Unsupported KernArgsVersion");

            bool handwritten = false;
            auto const* custom = config.find("CustomKernel");
            if(custom && custom->object().type == object_type::MAP
               && Dict(*custom).truthy("name", false))
                handwritten = !Dict(*custom).truthy("generated", false);
            else
                handwritten = config.truthy("CustomKernelName", false);

            if(version == 1 && !policy.persistentDataParallel()
               && (!regenerate || handwritten || config.has("InternalSupportParams")))
                throw std::runtime_error(
                    "PersistentLoopArgsVersion=1 requires DataParallel/StaticGrid");
            if(version == 1 && outerVersion != 3 && (!regenerate || handwritten))
                throw std::runtime_error("PersistentLoopArgsVersion=1 requires KernArgsVersion=3");
            return rv;
        }

        // ExecutionPolicy._policy, behind isPersistent and isStreamK.
        ExecutionPolicy policyOf(Dict const& state)
        {
            if(!state.has("TileProcessingStrategy")
               && (state.has("StreamK") || state.has("StreamKForceDPOnly")))
                return normalizeExecutionPolicy(state, true).policy;
            PolicySelectors selectors;
            if(auto const* value = state.find("TileProcessingStrategy"))
                selectors.strategy = value->text();
            if(auto const* value = state.find("WorkAssignment"))
                selectors.assignment = value->text();
            selectors.persistentLoop = state.find("_PersistentLoop");
            return resolvePolicy(selectors);
        }

        // Contractions.ProblemType, with what its predicates read beyond the record.
        struct ProblemType
        {
            ContractionSolution::ProblemType record;
            std::string                      activation;
            bool                             batched = false;
        };

        [[noreturn]] void inconsistentIndices()
        {
            throw std::runtime_error(
                "Solution state ProblemType index assignments are inconsistent");
        }

        // The only reconstruction of a problem type's index layout. Python builds
        // its index objects from problemIndexLayout and does not repeat this.
        ProblemIndexLayout indexLayout(Dict const& d)
        {
            struct Index
            {
                enum Kind
                {
                    Free,
                    Batch,
                    Bound
                } kind;
                int64_t                problemIndex = 0;
                int64_t                kindOrder    = 0;
                bool                   isA          = false;
                int64_t                i            = 0;
                int64_t                c            = 0;
                int64_t                d            = 0;
                std::optional<int64_t> a;
                std::optional<int64_t> b;
                bool                   aMirror = false;
                bool                   bMirror = false;
            };

            auto contains = [](std::vector<int64_t> const& list, int64_t value) {
                return std::find(list.begin(), list.end(), value) != list.end();
            };
            auto position = [](std::vector<int64_t> const& list, int64_t value) {
                return int64_t(std::find(list.begin(), list.end(), value) - list.begin());
            };

            auto total       = d.integer("TotalIndices");
            auto summation   = d["IndicesSummation"].integers();
            auto mirrorA     = d.has("MirrorDimsA") ? d["MirrorDimsA"].integers()
                                                    : std::vector<int64_t>{};
            auto mirrorB     = d.has("MirrorDimsB") ? d["MirrorDimsB"].integers()
                                                    : std::vector<int64_t>{};
            auto numIndicesC = d.integer("NumIndicesC");
            auto batch       = d["IndicesBatch"].integers();
            auto free        = d["IndicesFree"].integers();
            auto assignA     = d["IndexAssignmentsA"].integers();
            auto assignB     = d["IndexAssignmentsB"].integers();
            if(total < 0 || numIndicesC < 0)
                inconsistentIndices();

            using IndexPtr = std::shared_ptr<Index>;
            std::vector<IndexPtr> indices(total);
            std::vector<IndexPtr> freeIndices, batchIndices, boundIndices;
            auto normalize = [&](int64_t i) {
                if(i < 0)
                    i += total;
                if(i < 0 || i >= total)
                    inconsistentIndices();
                return i;
            };
            auto slot = [&](int64_t i) -> IndexPtr& { return indices[normalize(i)]; };

            for(auto i : summation)
            {
                auto bound          = std::make_shared<Index>(Index{Index::Bound});
                bound->aMirror      = contains(mirrorA, i);
                bound->bMirror      = contains(mirrorB, i);
                bound->problemIndex = normalize(i);
                bound->kindOrder    = int64_t(boundIndices.size());
                slot(i)             = bound;
                boundIndices.push_back(bound);
            }
            for(int64_t i = 0; i < numIndicesC; i++)
            {
                auto index          = std::make_shared<Index>(Index{Index::Free});
                index->c = index->d = i;
                index->problemIndex = i;
                if(contains(batch, i))
                {
                    index->kind      = Index::Batch;
                    index->kindOrder = int64_t(batchIndices.size());
                    slot(i)          = index;
                    batchIndices.push_back(index);
                    continue;
                }
                if(!contains(free, i))
                    inconsistentIndices();
                if(contains(assignA, i))
                {
                    index->isA = true;
                    index->i   = position(assignA, i);
                }
                else if(contains(assignB, i))
                {
                    index->isA = false;
                    index->i   = position(assignB, i);
                }
                else
                {
                    throw std::runtime_error("free index " + std::to_string(i)
                                             + " not in ia or ib");
                }
                index->kindOrder = int64_t(freeIndices.size());
                slot(i)          = index;
                freeIndices.push_back(index);
            }
            for(size_t ia = 0; ia < assignA.size(); ia++)
            {
                auto& index = slot(assignA[ia]);
                if(!index)
                    inconsistentIndices();
                index->a = int64_t(ia);
            }
            for(size_t ib = 0; ib < assignB.size(); ib++)
            {
                auto& index = slot(assignB[ib]);
                if(!index)
                    inconsistentIndices();
                index->b = int64_t(ib);
            }
            for(auto const& index : indices)
                if(!index || (index->kind != Index::Free && (!index->a || !index->b)))
                    inconsistentIndices();

            static std::string const allIndexNames = "ijklmnopqrstuvwxyz";
            if(numIndicesC + int64_t(boundIndices.size()) > int64_t(allIndexNames.size()))
                inconsistentIndices();
            std::string aNames(assignA.size(), '_');
            std::string bNames(assignB.size(), '_');
            std::string cNames(size_t(numIndicesC), '_');
            std::string dNames   = allIndexNames.substr(0, numIndicesC);
            std::string sumNames = allIndexNames.substr(numIndicesC, boundIndices.size());

            auto name = [](std::string& names, int64_t i) -> char& {
                if(i < 0)
                    i += int64_t(names.size());
                if(i < 0 || i >= int64_t(names.size()))
                    inconsistentIndices();
                return names[i];
            };
            for(auto const& index : freeIndices)
            {
                name(index->isA ? aNames : bNames, index->i) = name(dNames, index->d);
                name(cNames, index->c)                       = name(dNames, index->d);
            }
            for(auto const& index : batchIndices)
            {
                char letter          = name(dNames, index->d);
                name(aNames, *index->a) = letter;
                name(bNames, *index->b) = letter;
                name(cNames, index->c)  = letter;
            }
            for(size_t n = 0; n < boundIndices.size(); n++)
            {
                auto const& index = *boundIndices[n];
                if(!index.a || !index.b)
                    inconsistentIndices();
                char letter = sumNames[n];
                char upper  = char(std::toupper(static_cast<unsigned char>(letter)));
                name(aNames, *index.a) = index.aMirror ? upper : letter;
                name(bNames, *index.b) = index.bMirror ? upper : letter;
            }

            ProblemIndexLayout layout;
            layout.aNames   = aNames;
            layout.bNames   = bNames;
            layout.cNames   = cNames;
            layout.dNames   = dNames;
            layout.sumNames = sumNames;
            layout.aDims    = int64_t(assignA.size());
            layout.bDims    = int64_t(assignB.size());
            layout.cDims    = numIndicesC;
            layout.operationIdentifier = "Contraction_" + sumNames + "_A" + aNames
                                         + (d.truthy("ComplexConjugateA") ? "C" : "") + "_B"
                                         + bNames + (d.truthy("ComplexConjugateB") ? "C" : "")
                                         + "_C" + cNames + "_D" + dNames;
            layout.indices.reserve(indices.size());
            for(auto const& index : indices)
            {
                ProblemIndexSlot slot;
                if(index->kind == Index::Free)
                    slot.kind = 0;
                else if(index->kind == Index::Batch)
                    slot.kind = 1;
                else
                    slot.kind = 2;
                slot.problemIndex = index->problemIndex;
                slot.kindOrder    = index->kindOrder;
                slot.isA          = index->isA;
                slot.i            = index->i;
                slot.c            = index->c;
                slot.d            = index->d;
                slot.hasA         = index->a.has_value();
                slot.hasB         = index->b.has_value();
                slot.a            = slot.hasA ? *index->a : 0;
                slot.b            = slot.hasB ? *index->b : 0;
                slot.aMirror      = index->aMirror;
                slot.bMirror      = index->bMirror;
                layout.indices.push_back(slot);
            }
            return layout;
        }

        // Contractions.ProblemType.FromOriginalState.
        ProblemType problemTypeFromState(Dict const& d)
        {
            ProblemType rv;
            auto&       record = rv.record;

            auto opIdentifier = indexLayout(d).operationIdentifier;

            auto srcType     = d.dataType("DataType");
            auto dstType     = d.has("DestDataType") ? d.dataType("DestDataType") : srcType;
            auto computeType = d.has("ComputeDataType") ? d.dataType("ComputeDataType") : dstType;

            record.transA = d.truthy("TransposeA");
            record.transB = d.truthy("TransposeB");

            record.computeInputTypeA = d.has("MacDataTypeA") ? d.dataType("MacDataTypeA") : srcType;
            record.computeInputTypeB = d.has("MacDataTypeB") ? d.dataType("MacDataTypeB") : srcType;
            record.aType = d.has("DataTypeA") ? d.dataType("DataTypeA") : record.computeInputTypeA;
            record.bType = d.has("DataTypeB") ? d.dataType("DataTypeB") : record.computeInputTypeB;

            auto either = [&](rocisa::DataType type) {
                return record.aType == type || record.bType == type;
            };
            if(either(rocisa::DataType::Float8BFloat8))
            {
                record.aType = rocisa::DataType::Float8;
                record.bType = rocisa::DataType::BFloat8;
            }
            else if(either(rocisa::DataType::Float8BFloat8_fnuz))
            {
                record.aType = rocisa::DataType::Float8_fnuz;
                record.bType = rocisa::DataType::BFloat8_fnuz;
            }
            else if(either(rocisa::DataType::BFloat8Float8))
            {
                record.aType = rocisa::DataType::BFloat8;
                record.bType = rocisa::DataType::Float8;
            }
            else if(either(rocisa::DataType::BFloat8Float8_fnuz))
            {
                record.aType = rocisa::DataType::BFloat8_fnuz;
                record.bType = rocisa::DataType::Float8_fnuz;
            }

            record.eType = d.has("DataTypeE") ? d.dataType("DataTypeE") : dstType;
            if(d.has("DataTypeAmaxD"))
                d.dataType("DataTypeAmaxD");

            record.cType       = dstType;
            record.dType       = dstType;
            record.computeType = computeType;

            record.highPrecisionAccumulate = d.truthy("HighPrecisionAccumulate", false);
            record.useInitialStridesAB     = d.truthy("UseInitialStridesAB", false);
            record.useInitialStridesCD     = d.truthy("UseInitialStridesCD", false);
            record.stridedBatched          = d.truthy("StridedBatched", true);
            record.groupedGemm             = d.truthy("GroupedGemm", false);
            record.useBeta                 = d.truthy("UseBeta", true);

            record.useBias = 0;
            if(d.has("UseBias"))
            {
                record.useBias = d.integer("UseBias");
                record.biasDataTypeWhiteList = d.has("BiasDataTypeList")
                                                   ? sortedDataTypes(d["BiasDataTypeList"])
                                                   : biasDataTypeListDefault(d);
                if(auto const* biasSrc = d.find("BiasSrc"))
                {
                    static char const* const sources[] = {"A", "B", "C", "D"};
                    auto found = std::find_if(std::begin(sources),
                                              std::end(sources),
                                              [&](char const* s) { return biasSrc->isText(s); });
                    if(found == std::end(sources))
                        biasSrc->fail("one of A, B, C and D");
                    record.biasSrcWhiteList = {int(found - std::begin(sources))};
                }
            }

            record.useE = d.truthy("UseE", false);

            record.useGateResidual = false;
            if(d.has("UseGateResidual"))
            {
                record.useGateResidual = d.truthy("UseGateResidual");
                record.gateResidualDataTypeWhiteList
                    = d.has("GateResidualDataTypeList")
                          ? sortedDataTypes(d["GateResidualDataTypeList"])
                          : gateResidualDataTypeListDefault(d);
            }

            record.useGradient      = d.truthy("Gradient", false);
            record.outputAmaxD      = d.truthy("OutputAmaxD", false);
            record.useScaleAB       = d.has("UseScaleAB") ? d.text("UseScaleAB") : "";
            record.useScaleCD       = d.truthy("UseScaleCD", false);
            record.useScaleAlphaVec = d.integer("UseScaleAlphaVec", 0);

            rv.batched = d.truthy("Batched");

            rv.activation = d.has("ActivationType") ? activationName(d["ActivationType"]) : "none";
            record.activationType            = activationType(rv.activation);
            record.activationArgLength       = additionalArgNum(rv.activation);
            record.activationComputeDataType = d.dataType("ActivationComputeDataType");
            record.activationNoGuard         = d.truthy("ActivationNoGuard", false);
            record.sparse                    = d.integer("Sparse", 0);

            record.f32XdlMathOp
                = d.has("F32XdlMathOp") ? d.dataType("F32XdlMathOp") : rocisa::DataType::Float;

            record.supportDeviceUserArguments = d.truthy("SupportUserArgs", false);

            record.swizzleTensorA = d.truthy("SwizzleTensorA", false);
            record.swizzleTensorB = d.truthy("SwizzleTensorB", false);
            record.fusedGemmA2A   = d.truthy("FusedGemmA2A", false);

            record.mxBlockA = d.integer("MXBlockA", 0);
            record.mxBlockB = d.integer("MXBlockB", 0);
            record.mxTypeA
                = d.has("DataTypeMXSA") ? d.dataType("DataTypeMXSA") : rocisa::DataType::Float;
            record.mxTypeB
                = d.has("DataTypeMXSB") ? d.dataType("DataTypeMXSB") : rocisa::DataType::Float;

            record.metadataLayout = d.integer("MetadataLayout", 0);

            record.operationIdentifier = opIdentifier;
            return rv;
        }

        // Contractions.ProblemType.predicates(True, True, True).
        std::vector<std::shared_ptr<ProblemPredicate>> problemTypePredicates(ProblemType const& pt)
        {
            using namespace Predicates::Contraction;
            auto const& r = pt.record;

            std::vector<std::shared_ptr<ProblemPredicate>> rv;
            rv.push_back(withValue<OperationIdentifierEqual>(r.operationIdentifier));
            if(!r.useBeta)
                rv.push_back(std::make_shared<BetaZero>());
            rv.push_back(withValue<BiasDataTypeWhiteList>(r.biasDataTypeWhiteList));
            rv.push_back(withValue<GateResidualDataTypeWhiteList>(r.gateResidualDataTypeWhiteList));
            rv.push_back(withValue<BiasSrcWhiteList>(r.biasSrcWhiteList));
            rv.push_back(withValue<AmaxDCheck>(r.outputAmaxD));
            if(isAllActivations(pt.activation))
                rv.push_back(withValue<ActivationEnumWhiteList>(activationEnumList(
                    r.activationComputeDataType, pt.activation == "all", r.useGradient)));

            rv.push_back(withValue<TypesEqual>(std::array<rocisa::DataType, 6>{
                r.aType, r.bType, r.cType, r.dType, r.computeInputTypeA, r.computeInputTypeB}));
            rv.push_back(withValue<HighPrecisionAccumulateEqual>(r.highPrecisionAccumulate));
            rv.push_back(withValue<ActivationCheck>(r.activationType));
            rv.push_back(withValue<ActivationComputeTypeEqual>(r.activationComputeDataType));
            rv.push_back(withValue<ActivationNoGuardEqual>(r.activationNoGuard));
            rv.push_back(withValue<UseGradientEqual>(r.useGradient));
            rv.push_back(withValue<UseBiasCheck>(r.useBias));
            rv.push_back(withValue<UseEEqual>(r.useE));
            rv.push_back(withValue<UseGateResidualCheck>(r.useGateResidual));
            rv.push_back(withValue<DataTypeEEqual>(r.eType));
            rv.push_back(withValue<StridedBatchedEqual>(r.stridedBatched));
            rv.push_back(withValue<GroupedGemmEqual>(r.groupedGemm));
            rv.push_back(withValue<UseScaleABCheck>(r.useScaleAB));
            rv.push_back(withValue<UseScaleCDCheck>(r.useScaleCD));
            rv.push_back(withValue<UseScaleAlphaVecCheck>(r.useScaleAlphaVec));
            rv.push_back(withValue<Sparse>(r.sparse));
            rv.push_back(withValue<F32XdlMathOpEqual>(r.f32XdlMathOp));
            rv.push_back(withValue<SupportDeviceUserArguments>(r.supportDeviceUserArguments));
            rv.push_back(withValue<SwizzleTensorA>(r.swizzleTensorA));
            rv.push_back(withValue<SwizzleTensorB>(r.swizzleTensorB));
            rv.push_back(withValue<FusedGemmA2A>(r.fusedGemmA2A));
            if(r.fusedGemmA2A && pt.batched)
                rv.push_back(withIndexValue<BatchSizeEqual>(size_t(0), size_t(1)));
            rv.push_back(withValue<MXBlockA>(r.mxBlockA));
            if(r.mxBlockA)
                rv.push_back(withValue<DataTypeMXSA>(r.mxTypeA));
            rv.push_back(withValue<MXBlockB>(r.mxBlockB));
            if(r.mxBlockB)
                rv.push_back(withValue<DataTypeMXSB>(r.mxTypeB));
            return rv;
        }

        // Contractions.ASSERT_DIM_MAP_PREDICATES. A value of -1 skips that index.
        bool isDimAssertKey(std::string const& key)
        {
            return key == "AssertSizeEqual" || key == "AssertSizeGreaterThan"
                   || key == "AssertSizeLessThan" || key == "AssertStrideAEqual"
                   || key == "AssertStrideBEqual" || key == "AssertStrideCEqual"
                   || key == "AssertStrideDEqual";
        }

        // Contractions.extractDimPredicate. One constraint is that predicate;
        // more than one is their conjunction. An empty map, or a map whose
        // values are all -1, contributes nothing.
        std::shared_ptr<ProblemPredicate> dimAssertPredicate(std::string const& key,
                                                             Value const&       value)
        {
            using namespace Predicates::Contraction;
            if(value.object().type != object_type::MAP)
                throw std::runtime_error(key + " must be a dict of {index: value}");

            auto make = [&](size_t index, size_t size) -> std::shared_ptr<ProblemPredicate> {
                if(key == "AssertSizeEqual")
                    return withIndexValue<SizeEqual>(index, size);
                if(key == "AssertSizeGreaterThan")
                    return withIndexValue<SizeGreaterThan>(index, size);
                if(key == "AssertSizeLessThan")
                    return withIndexValue<SizeLessThan>(index, size);
                if(key == "AssertStrideAEqual")
                    return withIndexValue<StrideAEqual>(index, size);
                if(key == "AssertStrideBEqual")
                    return withIndexValue<StrideBEqual>(index, size);
                if(key == "AssertStrideCEqual")
                    return withIndexValue<StrideCEqual>(index, size);
                if(key == "AssertStrideDEqual")
                    return withIndexValue<StrideDEqual>(index, size);
                throw std::runtime_error("Unknown assertion key: " + key);
            };

            std::vector<std::shared_ptr<ProblemPredicate>> predicates;
            auto const&                                    map = value.object().via.map;
            for(uint32_t i = 0; i < map.size; i++)
            {
                Value index(map.ptr[i].key, value.path() + " index");
                Value entry(map.ptr[i].val, value.path());
                if(index.object().type != object_type::POSITIVE_INTEGER
                   && index.object().type != object_type::NEGATIVE_INTEGER)
                    index.fail("an integer index");
                auto position = index.integer();
                if(position < 0)
                    index.fail("an index >= 0");
                if(entry.equals(-1))
                    continue;
                if(entry.object().type != object_type::POSITIVE_INTEGER
                   && entry.object().type != object_type::NEGATIVE_INTEGER)
                    entry.fail("an integer");
                auto size = entry.integer();
                if(size < 0)
                    entry.fail("a size >= 0");
                predicates.push_back(make(size_t(position), size_t(size)));
            }
            if(predicates.empty())
                return nullptr;
            return conjunction(predicates);
        }

        // Contractions.ProblemPredicate.FromOriginalKeyPair.
        std::shared_ptr<ProblemPredicate> problemPredicateFromKeyPair(std::string const& key,
                                                                      Value const&       value)
        {
            using namespace Predicates::Contraction;
            if(key == "AssertAIGreaterThanEqual")
                return value.number() > 0 ? withValue<AIGreaterThanEqual>(value.number()) : nullptr;
            if(key == "AssertAILessThanEqual")
                return value.number() > 0 ? withValue<AILessThanEqual>(value.number()) : nullptr;
            if(isDimAssertKey(key))
                return dimAssertPredicate(key, value);

            auto endsWith = [&](std::string_view suffix) {
                return key.size() >= suffix.size()
                       && key.compare(key.size() - suffix.size(), suffix.size(), suffix) == 0;
            };
            if(endsWith("Multiple"))
            {
                if(value.equals(1))
                    return nullptr;
                if(key == "AssertFree0ElementMultiple")
                    return withIndexValue<Free0SizeMultiple>(size_t(0), size_t(value.integer()));
                if(key == "AssertFree1ElementMultiple")
                    return withIndexValue<Free1SizeMultiple>(size_t(0), size_t(value.integer()));
                if(key == "AssertSummationElementMultiple")
                    return withIndexValue<BoundSizeMultiple>(int64_t(-1), size_t(value.integer()));
                throw std::runtime_error("Unknown Multiple Value: " + key);
            }

            if(key.rfind("Assert", 0) == 0)
                throw std::runtime_error("Unknown assertion key: " + key);
            return nullptr;
        }

        // Contractions.ProblemPredicate.CompoundPredicates.
        std::vector<std::shared_ptr<ProblemPredicate>>
            compoundPredicates(Dict const& state, ProblemType const& problemType)
        {
            using namespace Predicates::Contraction;
            std::vector<std::shared_ptr<ProblemPredicate>> rv;
            auto const problem = state.dict("ProblemType");

            if(state.has("BatchSizeEqual"))
                rv.push_back(withIndexValue<BatchSizeEqual>(
                    size_t(0), size_t(state.integer("BatchSizeEqual"))));

            if(state.has("SynchronizerSizeCheck") && state.truthy("SynchronizerSizeCheck"))
            {
                std::array<int, 6> value;
                value[0] = int(state.integer("MacroTile0"));
                value[1] = int(state.integer("MacroTile1"));
                if(state.truthy("EnableMatrixInstruction"))
                {
                    auto miWaveTile = state["MIWaveTile"].list();
                    if(miWaveTile.size() < 2)
                        state["MIWaveTile"].fail("a list of 2 values");
                    value[2] = int(miWaveTile[0].integer() * miWaveTile[1].integer());
                }
                else
                    value[2] = int(state.integer("ThreadTile0") * state.integer("ThreadTile1"));
                if(!state["NumElementsPerBatchStore"].equals(0))
                    value[3] = int(quotient(state["NumElementsPerThread"],
                                            state["NumElementsPerBatchStore"]));
                else
                    value[3] = int(state.integer("NumElementsPerThread"));
                value[4] = int(std::ceil(quotient(state["NumThreads"], state["WavefrontSize"])));
                value[5] = int(state.integer("GlobalSplitU"));
                rv.push_back(withIndexValue<SynchronizerSizeCheck>(size_t(0), value));
            }

            if(state.dict("InternalSupportParams").integer("KernArgsVersion") >= 1
               && !policyOf(state).persistent())
                rv.push_back(withIndexValue<WorkgroupNumberCheck>(
                    size_t(0),
                    std::array<int, 3>{int(state.integer("MacroTile0")),
                                       int(state.integer("MacroTile1")),
                                       int(state.integer("GlobalSplitU"))}));

            if(problemType.record.aType != rocisa::DataType::Int8x4)
            {
                auto minFree0
                    = problem.truthy("TLUA") ? state.integer("GlobalReadVectorWidthA") : 1;
                auto minFree1
                    = problem.truthy("TLUB") ? state.integer("GlobalReadVectorWidthB") : 1;
                rv.push_back(withValue<LeadingFree0SizesGreaterOrEqual>(size_t(minFree0)));
                rv.push_back(withValue<LeadingFree1SizesGreaterOrEqual>(size_t(minFree1)));
            }

            if(state["PackedC0IndicesX"].list().size() > 1)
                rv.push_back(std::make_shared<CDStridesEqual>());

            if(state.has("KernelLanguage"))
                rv.push_back(withValue<KernelLanguageCompatible>(enumByName<KernelLanguage>(
                    state["KernelLanguage"],
                    static_cast<int>(KernelLanguage::Count),
                    [](KernelLanguage k) { return KernelLanguageTypeInfo::Get(k).name; },
                    "a kernel language")));

            if(state.has("GlobalSplitU")
               && (state["GlobalSplitU"].number() > 1 || state["GlobalSplitU"].equals(-1)))
            {
                if(!state.has("_GlobalAccumulation")
                   || !state["_GlobalAccumulation"].isText("MultipleBuffer"))
                    rv.push_back(withValue<DeterministicModeEqual>(false));
            }

            if(policyOf(state).streamK() && state.has("StreamKAtomic")
               && state["StreamKAtomic"].equals(1))
                rv.push_back(withValue<DeterministicModeEqual>(false));

            bool bufferLoad = state.has("BufferLoad") && state["BufferLoad"].equals(1);
            if(bufferLoad)
            {
                bool tlua       = problem.truthy("TLUA");
                bool tlub       = problem.truthy("TLUB");
                bool mayShiftA  = tlua
                                 && state["AssertFree0ElementMultiple"].number()
                                        < state["GlobalReadVectorWidthA"].number();
                bool mayShiftB  = tlub
                                 && state["AssertFree1ElementMultiple"].number()
                                        < state["GlobalReadVectorWidthB"].number();
                BufferLoadCheckPacket packet;
                packet.shiftPtrElemB = mayShiftB ? state.integer("GlobalReadVectorWidthB") : 0;
                packet.shiftPtrElemA = mayShiftA ? state.integer("GlobalReadVectorWidthA") : 0;
                packet.depthUorMT1   = tlub ? state.integer("DepthU") : state.integer("MacroTile1");
                packet.depthUorMT0   = tlua ? state.integer("DepthU") : state.integer("MacroTile0");
                rv.push_back(withValue<BufferLoadOffsetLimitCheck>(packet));
            }

            if(bufferLoad)
                rv.push_back(withValue<BufferLoadOffsetLimitCheck_Beta>(
                    size_t(state.integer("MacroTile1"))));

            if(state.has("BufferStore") && state["BufferStore"].equals(1))
                rv.push_back(
                    withValue<BufferStoreOffsetLimitCheck>(size_t(state.integer("MacroTile1"))));

            if(state.has("_GlobalAccumulation") && !state["_GlobalAccumulation"].isNone()
               && !policyOf(state).persistent())
                rv.push_back(withValue<GlobalSplitUCheckMinK>(
                    std::array<int, 2>{32, int(state.integer("GlobalSplitU"))}));

            if(state.has("WorkGroupMappingXCC") && state.has("WorkGroupMappingXCCGroup"))
            {
                // The library stores only `value`. Loading the file reads the
                // device compute-unit count; building this record must not.
                rv.push_back(std::make_shared<WorkgroupMappingXCCCheck>(
                    std::array<int, 2>{int(state.integer("WorkGroupMappingXCC")),
                                       int(state.integer("WorkGroupMappingXCCGroup"))},
                    size_t{0}));
            }

            if(problem.truthy("SwizzleTensorA"))
                rv.push_back(withValue<SwizzleTensorA>(true));

            if(problem.truthy("SwizzleTensorB"))
                rv.push_back(withValue<SwizzleTensorB>(true));

            if(problem.truthy("FusedGemmA2A"))
                rv.push_back(withValue<FusedA2ATileDivisible>(state.integer("MacroTile0")));

            if(state.truthy("ReuseAcrossPersistent", false))
            {
                auto kIdx       = size_t(problem.integer("NumIndicesC"));
                auto kTiles     = state.integer("_RAPNumResidentKTiles");
                auto depthU     = state.integer("DepthU");
                int64_t floorTiles = 1;
                rv.push_back(
                    withIndexValue<SizeEqual>(size_t(0), size_t(state.integer("MacroTile0"))));
                rv.push_back(
                    withIndexValue<SizeMultiple>(size_t(1), size_t(state.integer("MacroTile1"))));
                rv.push_back(withIndexValue<SizeMultiple>(kIdx, size_t(depthU)));
                rv.push_back(
                    withIndexValue<SizeGreaterThan>(kIdx, size_t(floorTiles * depthU - 1)));
                rv.push_back(withIndexValue<SizeLessThan>(kIdx, size_t(kTiles * depthU + 1)));
            }

            return rv;
        }

        // Contractions.ProblemPredicate.FromOriginalState.
        std::shared_ptr<ProblemPredicate> problemPredicateFromState(Dict const&        d,
                                                                    ProblemType const& problemType)
        {
            auto problemTypePreds = problemTypePredicates(problemType);
            auto compoundPreds    = compoundPredicates(d, problemType);

            std::vector<std::shared_ptr<ProblemPredicate>> predicates;
            for(auto const& item : d.items())
                if(auto predicate = problemPredicateFromKeyPair(item.first, item.second))
                    predicates.push_back(predicate);
            predicates.insert(predicates.end(), problemTypePreds.begin(), problemTypePreds.end());
            predicates.insert(predicates.end(), compoundPreds.begin(), compoundPreds.end());
            return conjunction(predicates);
        }

        // Contractions.TaskPredicate.FromOriginalState.
        std::shared_ptr<TaskPredicate> taskPredicateFromState(Dict const& d)
        {
            std::vector<std::shared_ptr<TaskPredicate>> predicates;
            for(auto const& item : d.items())
                if(item.first == "_WorkspaceSizePerElemC" && item.second.number() > 0)
                    predicates.push_back(
                        std::make_shared<Predicates::Contraction::WorkspaceCheck>());
            if(!policyOf(d).persistent())
                predicates.push_back(std::make_shared<Predicates::Contraction::LaunchLimits>());
            return conjunction(predicates);
        }

        // Hardware.HardwarePredicate._createPciChipIdPredicate.
        std::shared_ptr<AMDGPUPredicate> pciChipIdPredicate(Value const* deviceNames)
        {
            std::vector<Value> names;
            if(deviceNames && !deviceNames->isNone())
            {
                if(deviceNames->object().type == object_type::STR)
                    names.push_back(*deviceNames);
                else if(deviceNames->truthy())
                    names = deviceNames->list();
            }

            static std::regex const deviceName("^Device\\s+([0-9a-fA-F]+)$");
            std::vector<std::shared_ptr<AMDGPUPredicate>> predicates;
            for(auto const& name : names)
            {
                if(name.isNone())
                    continue;
                auto        text  = name.text();
                auto        first = text.find_first_not_of(" \t\n\r\f\v");
                auto        last  = text.find_last_not_of(" \t\n\r\f\v");
                std::string stripped
                    = first == std::string::npos ? "" : text.substr(first, last - first + 1);
                std::smatch match;
                if(!std::regex_match(stripped, match, deviceName))
                    throw std::runtime_error("Invalid device name format: '" + text
                                             + "', expected 'Device XXXX'");
                auto chipId = match[1].str();
                if(std::find(SupportedBuildChipIds.begin(),
                             SupportedBuildChipIds.end(),
                             Value::lower(chipId))
                   == SupportedBuildChipIds.end())
                    continue;
                predicates.push_back(std::make_shared<Predicates::GPU::PciChipIdEqual>(
                    int(std::stoll(chipId, nullptr, 16))));
            }
            if(predicates.empty())
                return nullptr;
            return disjunction(predicates);
        }

        // Hardware.HardwarePredicate.FromHardware.
        std::shared_ptr<HardwarePredicate>
            hardwarePredicate(Value const& isaValue, Value const* cuCount, Value const* deviceNames)
        {
            auto isa = isaValue.list();
            if(isa.size() < 3)
                isaValue.fail("an ISA version");
            auto               stepping = isa[2].integer();
            std::ostringstream step;
            if(stepping < 0)
                step << '-';
            step << std::hex << (stepping < 0 ? -uint64_t(stepping) : uint64_t(stepping));
            auto gfxArch = "gfx" + std::to_string(isa[0].integer())
                           + std::to_string(isa[1].integer()) + step.str();

            auto processor = AMDGPU::toProcessor(gfxArch);
            if(AMDGPU::toString(processor) != gfxArch)
                throw std::runtime_error("Solution state ISA names an unknown processor "
                                         + gfxArch);

            std::vector<std::shared_ptr<AMDGPUPredicate>> props;
            props.push_back(std::make_shared<Predicates::GPU::ProcessorEqual>(processor));
            if(cuCount && !cuCount->isNone())
                props.push_back(
                    std::make_shared<Predicates::GPU::CUCountEqual>(int(cuCount->integer())));
            if(gfxArch == "gfx950")
                if(auto pciChipId = pciChipIdPredicate(deviceNames))
                    props.push_back(pciChipId);

            return std::make_shared<Predicates::IsSubclass<Hardware, AMDGPU>>(
                props.size() == 1 ? props.front() : conjunction(props));
        }

        // Contractions.SizeMapping.FromOriginalState.
        SizeMapping sizeMappingFromState(Dict const& d, SizeMapping s)
        {
            auto normalized = normalizeExecutionPolicy(d, false);

            int globalAccum = 0;
            auto const& accumulation = d["_GlobalAccumulation"];
            if(accumulation.isText("SingleBuffer"))
                globalAccum = 1;
            if(accumulation.isText("MultipleBuffer"))
                globalAccum = 2;
            if(accumulation.isText("MultipleBufferSingleKernel"))
                globalAccum = 3;
            if(accumulation.isText("PartialsBuffer"))
                globalAccum = 4;
            auto pgr = int64_t(d["PrefetchGlobalRead"].number());

            double tile;
            if(d.truthy("EnableMatrixInstruction"))
            {
                auto miWaveTile = d["MIWaveTile"].list();
                if(miWaveTile.size() < 2)
                    d["MIWaveTile"].fail("a list of 2 values");
                tile = miWaveTile[0].number() * miWaveTile[1].number();
            }
            else
                tile = d["ThreadTile0"].number() * d["ThreadTile1"].number();
            double elements = !d["NumElementsPerBatchStore"].equals(0)
                                  ? std::ceil(quotient(d["NumElementsPerThread"],
                                                       d["NumElementsPerBatchStore"]))
                                  : d["NumElementsPerThread"].number();
            double waves = std::ceil(quotient(d["NumThreads"], d["WavefrontSize"]));
            auto   synchronizerSizePerWG = std::ceil(tile * elements * waves);

            auto workGroupMapping = [&]() -> int {
                if(d["SpaceFillingAlgo"].list().empty())
                    return int(d.integer("WorkGroupMapping"));
                // convertSFCWGMListToHex: [[M1, N1], [M2, N2]] packed from the
                // low byte up as M1, N1, M2, N2, read back as a signed int.
                uint64_t output = 0;
                auto     levels = d["SFCWGM"].list();
                for(size_t idx = 0; idx < levels.size(); idx++)
                {
                    auto dims = levels[idx].list();
                    if(dims.size() < 2)
                        levels[idx].fail("a list of 2 values");
                    if(16 * idx < 32)
                        output |= uint64_t(dims[0].integer()) << (16 * idx);
                    if(8 + 16 * idx < 32)
                        output |= uint64_t(dims[1].integer()) << (8 + 16 * idx);
                }
                return static_cast<int32_t>(static_cast<uint32_t>(output & 0xFFFFFFFF));
            };

            bool dtva = d.truthy("DirectToVgprA");
            bool dtvb = d.truthy("DirectToVgprB");
            bool dtlA = d.truthy("DirectToLdsA");
            bool dtlB = d.truthy("DirectToLdsB");

            s.waveNum = floorQuotient(d["NumThreads"], d["WavefrontSize"]);
            assignSequence(d["WorkGroup"], s.workGroupSize);
            s.macroTile = dim3(d.integer("MacroTile0"), d.integer("MacroTile1"), 1);
            assignSequence(d["ThreadTile"], s.threadTile);
            assignArray(d["MatrixInstruction"], s.matrixInstruction);
            s.grvwA              = d.integer("GlobalReadVectorWidthA");
            s.grvwB              = d.integer("GlobalReadVectorWidthB");
            s.gwvwC              = d.integer("StoreVectorWidth");
            s.gwvwD              = d.integer("StoreVectorWidth");
            s.workGroupMapping   = workGroupMapping();
            s.staggerU           = d.integer("StaggerU", 0);
            s.staggerUMapping    = d.integer("StaggerUMapping", 0);
            s.depthU             = d.integer("DepthU");
            s.globalSplitUPGR    = GlobalSplitUPGR;
            s.globalSplitU       = int16_t(d.integer("GlobalSplitU"));
            s.staggerStrideShift = d.integer("_staggerStrideShift", 0);
            s.packBatchDims      = 0;

            s.tileProcessingStrategy   = normalized.policy.strategy;
            s.workAssignment           = normalized.policy.assignment;
            s.streamKAtomic            = int(normalized.streamKAtomic);
            s.prefetchAcrossPersistent = int(normalized.prefetchAcrossPersistent);
            s.magicDivAlg              = int(d.integer("MagicDivAlg", 1));
            s.sourceKernel             = d["KernelLanguage"].isText("Source");
            s.globalAccumulation       = globalAccum;
            s.adaptiveGemmGSUA         = int(d.integer("AdaptiveGemmGSUA", 0));
            s.workspaceSizePerElemC    = d.integer("_WorkspaceSizePerElemC");
            s.workspaceSizePerElemBias = d.integer("_WorkspaceSizePerElemBias");
            s.activationFused          = d.truthy("ActivationFused");
            s.workGroupMappingXCC      = int(d.integer("WorkGroupMappingXCC"));
            s.workGroupMappingXCCGroup = int(d.integer("WorkGroupMappingXCCGroup"));
            s.globalSplitUCoalesced    = d.truthy("GlobalSplitUCoalesced");
            s.globalSplitUWorkGroupMappingRoundRobin
                = d.truthy("GlobalSplitUWorkGroupMappingRoundRobin");

            s.CUOccupancy              = int(d.integer("CUOccupancy"));
            s.PrefetchGlobalRead       = int(pgr);
            s.MathClocksUnrolledLoop   = int(d.integer("MathClocksUnrolledLoop"));
            s.synchronizerSizePerWG    = size_t(synchronizerSizePerWG);
            s.nonTemporalA             = int(d.integer("NonTemporalA"));
            s.nonTemporalB             = int(d.integer("NonTemporalB"));
            s.temporalHintA            = int(d.integer("TemporalHintA", 0));
            s.temporalHintB            = int(d.integer("TemporalHintB", 0));
            s.hasTemporalHint          = d.truthy("_HasTemporalHint", false);
            s.adaptiveGemmNTAB         = int(d.integer("AdaptiveGemmNTAB", 0));
            s.customMainLoopScheduling = int(d.integer("UseCustomMainLoopSchedule"));
            s.useSubtileImpl           = d.truthy("UseSubtileImpl", false);

            s.NonTemporalD                  = int(d.integer("NonTemporalD"));
            s.WaveSeparateGlobalReadA       = int(d.integer("WaveSeparateGlobalReadA"));
            s.WaveSeparateGlobalReadB       = int(d.integer("WaveSeparateGlobalReadB"));
            s.UnrollLoopSwapGlobalReadOrder = int(d.integer("UnrollLoopSwapGlobalReadOrder"));
            s.DirectToVgprA                 = dtva;
            s.DirectToVgprB                 = dtvb;
            s.NumLoadsCoalescedA            = int(d.integer("NumLoadsCoalescedA"));
            s.NumLoadsCoalescedB            = int(d.integer("NumLoadsCoalescedB"));
            assignArray(d["MIWaveGroup"], s.waveGroup);
            s.VectorWidthA         = int(d.integer("VectorWidthA"));
            s.VectorWidthB         = int(d.integer("VectorWidthB"));
            s.LocalSplitU          = int(d.integer("LocalSplitU"));
            s.DirectToLdsA         = dtlA;
            s.DirectToLdsB         = dtlB;
            s.expertSchedulingMode = int(d.integer("ExpertSchedulingMode", 0));
            assignSequence(d["ClusterDim"], s.clusterDim);
            s.validateExecutionPolicy();
            return s;
        }

        // Contractions.InternalArgsSupport.FromOriginalState.
        ContractionSolution::InternalArgsSupport internalArgsSupportFromState(Dict const& d)
        {
            auto isp    = d.dict("InternalSupportParams");
            bool useSFC = isp.truthy("UseSFC")
                          || (d.has("SpaceFillingAlgo") && !d["SpaceFillingAlgo"].list().empty());
            ContractionSolution::InternalArgsSupport rv;
            rv.version                   = isp.integer("KernArgsVersion");
            rv.persistentLoopArgsVersion = isp.integer("PersistentLoopArgsVersion", 0);
            rv.gsu                       = isp.truthy("SupportUserGSU");
            rv.wgm                       = isp.truthy("SupportCustomWGM");
            rv.staggerU                  = isp.truthy("SupportCustomStaggerU");
            rv.perTileExtraIters         = isp.truthy("SupportStreamKPerTileExtraIters", false);
            rv.useUniversalArgs          = isp.truthy("UseUniversalArgs");
            rv.useSFC                    = useSFC;
            return rv;
        }

        // Contractions.CustomKernel.FromOriginalState, read as the library
        // reads the record's customKernel.
        CustomKernel customKernelFromState(Dict const& d)
        {
            CustomKernel rv;
            rv.name = d.text("name");
            for(auto const& argValue : d["args"].list())
            {
                Dict                arg(argValue);
                CustomArgDefinition definition;
                definition.type = enumByName<CustomArgType>(
                    arg["type"],
                    CustomArgType_Count,
                    [](CustomArgType t) { return toString(t); },
                    "a custom argument type");
                definition.semantic = enumByName<CustomArgSemantic>(
                    arg["semantic"],
                    static_cast<int>(CustomArgSemantic::COUNT),
                    [](CustomArgSemantic t) { return toString(t); },
                    "a custom argument semantic");
                definition.padding = arg.integer("padding", 0);
                definition.index   = arg.integer("index", 0);
                rv.args.push_back(definition);
            }
            assignSequence(d["macrotile"], rv.macrotile);
            assignSequence(d["threads"], rv.threads);
            auto grid = d["grid"].list();
            if(grid.size() > 3)
                d["grid"].fail("a list of at most 3 values");
            CustomGridSize* gridSizes[] = {&rv.grid.x, &rv.grid.y, &rv.grid.z};
            for(size_t i = 0; i < grid.size(); i++)
                *gridSizes[i] = enumByName<CustomGridSize>(
                    grid[i],
                    CustomGridSize_Count,
                    [](CustomGridSize t) { return toString(t); },
                    "a custom grid size");
            if(d.has("workspaceType"))
                rv.workspaceType = enumByName<CustomWorkspaceType>(
                    d["workspaceType"],
                    CustomWorkspaceType_Count,
                    [](CustomWorkspaceType t) { return toString(t); },
                    "a custom workspace type");
            rv.workspaceSizePerElemC    = d.integer("workspaceSizePerElemC", 0);
            rv.workspaceSizePerElemBias = d.integer("workspaceSizePerElemBias", 0);
            rv.generated                = d.truthy("generated", false);
            return rv;
        }
    } // namespace

    ProblemIndexLayout problemIndexLayout(msgpack::object const& problemType)
    {
        return indexLayout(Dict(Value(problemType, "ProblemType")));
    }

    std::shared_ptr<ContractionSolution> solutionFromState(msgpack::object const& state)
    {
        Dict d{Value(state, "")};
        auto rv = std::make_shared<ContractionSolution>();

        if(d.has("SolutionNameMin"))
            rv->solutionName = d.text("SolutionNameMin");
        if(d.has("KernelNameMin"))
            rv->kernelName = d.text("KernelNameMin");

        auto problemType = problemTypeFromState(d.dict("ProblemType"));
        rv->problemType  = problemType.record;

        int mxScaleFormat = 0;
        if(auto const* value = d.find("MXScaleFormat"))
        {
            if(value->isText("HostPreSwizzle"))
                mxScaleFormat = 1;
            else if(value->isText("InMemorySwizzle"))
                mxScaleFormat = 2;
        }
        rv->problemType.mxScaleFormat = mxScaleFormat;

        rv->problemPredicate = problemPredicateFromState(d, problemType);
        rv->taskPredicate    = taskPredicateFromState(d);

        if(d.has("DebugKernel"))
            rv->debugKernel = d.truthy("DebugKernel");

        if(d.has("SolutionIndex"))
            rv->index = d.integer("SolutionIndex");

        rv->libraryLogicIndex = d.integer("SolutionIndex", -1);

        rv->sizeMapping = sizeMappingFromState(d, rv->sizeMapping);
        if(d.has("CustomKernel"))
            rv->customKernel = customKernelFromState(d.dict("CustomKernel"));

        rv->internalArgsSupport = internalArgsSupportFromState(d);
        rv->validatePersistentLoopArgs();

        if(auto const* ideals = d.find("Ideals"))
        {
            if(ideals->object().type != object_type::MAP)
                ideals->fail("a map");
            auto const& map = ideals->object().via.map;
            for(uint32_t i = 0; i < map.size; i++)
            {
                Value key(map.ptr[i].key, "Ideals key");
                Value value(map.ptr[i].val, "Ideals value");
                auto  index = key.object().type == object_type::STR ? std::stoi(key.text())
                                                                     : int(key.integer());
                rv->ideals[index] = value.number();
            }
        }

        if(auto const* linearModel = d.find("LinearModel"))
        {
            Dict model(*linearModel);
            if(auto const* value = model.find("slope"))
                rv->linearModel.slope = value->number();
            if(auto const* value = model.find("intercept"))
                rv->linearModel.intercept = value->number();
            if(auto const* value = model.find("max"))
                rv->linearModel.max = value->number();
        }

        rv->hardwarePredicate
            = hardwarePredicate(d["ISA"], d.find("CUCount"), d.find("DeviceNames"));

        return rv;
    }
} // namespace TensileLite
