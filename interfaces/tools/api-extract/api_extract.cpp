// Copyright Advanced Micro Devices, Inc.
// SPDX-License-Identifier: MIT
#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "clang/AST/ASTConsumer.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Mangle.h"
#include "clang/AST/RecordLayout.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Tooling/CommonOptionsParser.h"
#include "clang/Tooling/Tooling.h"
#include "llvm/ADT/APSInt.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

namespace {
llvm::cl::OptionCategory category("ROCm API extraction options");
llvm::cl::opt<std::string> output_path("output", llvm::cl::Required,
                                       llvm::cl::desc("Output JSON path"), llvm::cl::cat(category));
llvm::cl::opt<std::string> header_root("header-root", llvm::cl::Required,
                                       llvm::cl::desc("Only declarations below this path"),
                                       llvm::cl::cat(category));

struct Item {
    std::string key;
    llvm::json::Object value;
};

class Collector {
   public:
    void add(std::string key, llvm::json::Object value) {
        items_.push_back({std::move(key), std::move(value)});
    }

    bool write(llvm::StringRef path) {
        std::stable_sort(items_.begin(), items_.end(),
                         [](const Item& a, const Item& b) { return a.key < b.key; });
        llvm::json::Array declarations;
        for (Item& item : items_) declarations.push_back(std::move(item.value));
        llvm::json::Object root;
        root["schema_version"] = 1;
        root["declarations"] = std::move(declarations);
        std::error_code error;
        llvm::raw_fd_ostream stream(path, error);
        if (error) {
            llvm::errs() << "cannot open " << path << ": " << error.message() << "\n";
            return false;
        }
        stream << llvm::formatv("{0:2}\n", llvm::json::Value(std::move(root)));
        return true;
    }

   private:
    std::vector<Item> items_;
};

class Visitor : public clang::RecursiveASTVisitor<Visitor> {
   public:
    Visitor(clang::ASTContext& context, Collector& collector)
        : context_(context),
          collector_(collector),
          mangle_context_(context.createMangleContext()) {}

    bool VisitFunctionDecl(clang::FunctionDecl* declaration) {
        if (!accept(declaration) || !declaration->isCanonicalDecl()) return true;
        llvm::json::Array parameters;
        for (const clang::ParmVarDecl* parameter : declaration->parameters()) {
            llvm::json::Object item;
            item["name"] = parameter->getNameAsString();
            item["type"] = parameter->getType().getAsString();
            parameters.push_back(std::move(item));
        }
        llvm::json::Object item = base(declaration, "function");
        item["type"] = declaration->getType().getAsString();
        item["return_type"] = declaration->getReturnType().getAsString();
        item["parameters"] = std::move(parameters);
        item["variadic"] = declaration->isVariadic();
        item["c_linkage"] = declaration->isExternC();
        item["inline_specified"] = declaration->isInlineSpecified();
        item["templated_kind"] = static_cast<int64_t>(declaration->getTemplatedKind());
        item["visibility"] = static_cast<int64_t>(declaration->getVisibility());
        std::string linkage_name;
        llvm::raw_string_ostream linkage_stream(linkage_name);
        if (mangle_context_->shouldMangleDeclName(declaration)) {
            if (const auto* constructor = llvm::dyn_cast<clang::CXXConstructorDecl>(declaration)) {
                mangle_context_->mangleName(clang::GlobalDecl(constructor, clang::Ctor_Complete),
                                            linkage_stream);
            } else if (const auto* destructor =
                           llvm::dyn_cast<clang::CXXDestructorDecl>(declaration)) {
                mangle_context_->mangleName(clang::GlobalDecl(destructor, clang::Dtor_Complete),
                                            linkage_stream);
            } else {
                mangle_context_->mangleName(clang::GlobalDecl(declaration), linkage_stream);
            }
        } else {
            linkage_stream << declaration->getName();
        }
        item["linkage_name"] = linkage_stream.str();
        add(declaration, std::move(item));
        return true;
    }

    bool VisitEnumDecl(clang::EnumDecl* declaration) {
        if (!accept(declaration) || !declaration->isCanonicalDecl() ||
            !declaration->isCompleteDefinition())
            return true;
        llvm::json::Array values;
        for (const clang::EnumConstantDecl* constant : declaration->enumerators()) {
            llvm::SmallString<32> buffer;
            constant->getInitVal().toString(buffer, 10);
            llvm::json::Object value;
            value["name"] = constant->getNameAsString();
            value["value"] = std::string(buffer);
            values.push_back(std::move(value));
        }
        llvm::json::Object item = base(declaration, "enum");
        item["underlying_type"] = declaration->getIntegerType().getAsString();
        item["values"] = std::move(values);
        add(declaration, std::move(item));
        return true;
    }

    bool VisitRecordDecl(clang::RecordDecl* declaration) {
        if (!accept(declaration) || !declaration->isCanonicalDecl() ||
            !declaration->isCompleteDefinition())
            return true;
        llvm::json::Array fields;
        for (const clang::FieldDecl* field : declaration->fields()) {
            llvm::json::Object value;
            value["name"] = field->getNameAsString();
            value["type"] = field->getType().getAsString();
            fields.push_back(std::move(value));
        }
        llvm::json::Object item = base(declaration, declaration->isUnion() ? "union" : "record");
        if (!declaration->isInvalidDecl() && !declaration->isDependentType()) {
            const clang::ASTRecordLayout& layout = context_.getASTRecordLayout(declaration);
            item["size_bits"] = static_cast<int64_t>(layout.getSize().getQuantity() * 8);
            item["align_bits"] = static_cast<int64_t>(layout.getAlignment().getQuantity() * 8);
            item["layout_available"] = true;
        } else {
            item["layout_available"] = false;
        }
        item["fields"] = std::move(fields);
        add(declaration, std::move(item));
        return true;
    }

    bool VisitTypedefNameDecl(clang::TypedefNameDecl* declaration) {
        if (!accept(declaration) || !declaration->isCanonicalDecl()) return true;
        llvm::json::Object item = base(declaration, "typedef");
        item["underlying_type"] = declaration->getUnderlyingType().getAsString();
        add(declaration, std::move(item));
        return true;
    }

   private:
    bool accept(const clang::Decl* declaration) const {
        const clang::SourceManager& source_manager = context_.getSourceManager();
        clang::SourceLocation location = source_manager.getExpansionLoc(declaration->getLocation());
        if (location.isInvalid() || source_manager.isInSystemHeader(location)) return false;
        llvm::StringRef filename = source_manager.getFilename(location);
        return filename.starts_with(header_root);
    }

    llvm::json::Object base(const clang::NamedDecl* declaration, llvm::StringRef kind) const {
        const clang::SourceManager& source_manager = context_.getSourceManager();
        clang::PresumedLoc location = source_manager.getPresumedLoc(declaration->getLocation());
        std::string file;
        if (location.isValid()) {
            llvm::StringRef filename = location.getFilename();
            if (filename.consume_front(header_root)) {
                filename = filename.ltrim("/\\");
            }
            file = filename.str();
        }
        llvm::json::Object item;
        item["kind"] = kind;
        item["name"] = stable_name(declaration);
        item["file"] = file;
        item["line"] = location.isValid() ? static_cast<int64_t>(location.getLine()) : 0;
        return item;
    }

    void add(const clang::NamedDecl* declaration, llvm::json::Object item) {
        collector_.add(stable_name(declaration) + ":" +
                           std::to_string(declaration->getLocation().getRawEncoding()),
                       std::move(item));
    }

    std::string stable_name(const clang::NamedDecl* declaration) const {
        std::string name = declaration->getQualifiedNameAsString();
        for (const char separator : {'/', '\\'}) {
            const std::string prefix = header_root.getValue() + separator;
            size_t position = 0;
            while ((position = name.find(prefix, position)) != std::string::npos) {
                name.erase(position, prefix.size());
            }
        }
        return name;
    }

    clang::ASTContext& context_;
    Collector& collector_;
    std::unique_ptr<clang::MangleContext> mangle_context_;
};

class Consumer : public clang::ASTConsumer {
   public:
    Consumer(clang::ASTContext& context, Collector& collector) : visitor_(context, collector) {}
    void HandleTranslationUnit(clang::ASTContext& context) override {
        visitor_.TraverseDecl(context.getTranslationUnitDecl());
    }

   private:
    Visitor visitor_;
};

class Action : public clang::ASTFrontendAction {
   public:
    explicit Action(Collector& collector) : collector_(collector) {}
    std::unique_ptr<clang::ASTConsumer> CreateASTConsumer(clang::CompilerInstance& compiler,
                                                          llvm::StringRef) override {
        return std::make_unique<Consumer>(compiler.getASTContext(), collector_);
    }

   private:
    Collector& collector_;
};

class Factory : public clang::tooling::FrontendActionFactory {
   public:
    explicit Factory(Collector& collector) : collector_(collector) {}
    std::unique_ptr<clang::FrontendAction> create() override {
        return std::make_unique<Action>(collector_);
    }

   private:
    Collector& collector_;
};
}  // namespace

int main(int argc, const char** argv) {
    auto parser = clang::tooling::CommonOptionsParser::create(argc, argv, category);
    if (!parser) {
        llvm::errs() << parser.takeError();
        return 2;
    }
    Collector collector;
    clang::tooling::ClangTool tool(parser->getCompilations(), parser->getSourcePathList());
    Factory factory(collector);
    int result = tool.run(&factory);
    if (result != 0) return result;
    return collector.write(output_path) ? 0 : 1;
}
