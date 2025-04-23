/*
Copyright (c) 2019 - 2023 Advanced Micro Devices, Inc. All rights reserved.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/
#pragma once
#include <rapidjson/document.h>
#include <rapidjson/reader.h>

#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <string>

RAPIDJSON_DIAG_PUSH
#ifdef __GNUC__
RAPIDJSON_DIAG_OFF(effc++)
#endif

// This example demonstrates JSON token-by-token parsing with an API that is
// more direct; you don't need to design your logic around a handler object and
// callbacks. Instead, you retrieve values from the JSON stream by calling
// GetInt(), GetDouble(), GetString() and GetBool(), traverse into structures
// by calling EnterObject() and EnterArray(), and skip over unwanted data by
// calling SkipValue(). When you know your JSON's structure, this can be quite
// convenient.
//
// If you aren't sure of what's next in the JSON data, you can use PeekType() and
// PeekValue() to look ahead to the next object before reading it.
//
// If you call the wrong retrieval method--e.g. GetInt when the next JSON token is
// not an int, EnterObject or EnterArray when there isn't actually an object or array
// to read--the stream parsing will end immediately and no more data will be delivered.
//
// After calling EnterObject, you retrieve keys via NextObjectKey() and values via
// the normal getters. When NextObjectKey() returns null, you have exited the
// object, or you can call SkipObject() to skip to the end of the object
// immediately. If you fetch the entire object (i.e. NextObjectKey() returned  null),
// you should not call SkipObject().
//
// After calling EnterArray(), you must alternate between calling NextArrayValue()
// to see if the array has more data, and then retrieving values via the normal
// getters. You can call SkipArray() to skip to the end of the array immediately.
// If you fetch the entire array (i.e. NextArrayValue() returned null),
// you should not call SkipArray().
//
// This parser uses in-situ strings, so the JSON buffer will be altered during the
// parse.

using namespace rapidjson;

class LookaheadParserHandler {
   public:
    bool Null() {
        st_ = kHasNull;
        v_.SetNull();
        return true;
    }
    bool Bool(bool b) {
        st_ = kHasBool;
        v_.SetBool(b);
        return true;
    }
    bool Int(int i) {
        st_ = kHasNumber;
        v_.SetInt(i);
        return true;
    }
    bool Uint(unsigned u) {
        st_ = kHasNumber;
        v_.SetUint(u);
        return true;
    }
    bool Int64(int64_t i) {
        st_ = kHasNumber;
        v_.SetInt64(i);
        return true;
    }
    bool Uint64(uint64_t u) {
        st_ = kHasNumber;
        v_.SetUint64(u);
        return true;
    }
    bool Double(double d) {
        st_ = kHasNumber;
        v_.SetDouble(d);
        return true;
    }
    bool RawNumber(const char*, SizeType, bool) { return false; }
    bool String(const char* str, SizeType length, bool) {
        st_ = kHasString;
        v_.SetString(str, length);
        return true;
    }
    bool StartObject() {
        st_ = kEnteringObject;
        return true;
    }
    bool Key(const char* str, SizeType length, bool) {
        st_ = kHasKey;
        v_.SetString(str, length);
        return true;
    }
    bool EndObject(SizeType) {
        st_ = kExitingObject;
        return true;
    }
    bool StartArray() {
        st_ = kEnteringArray;
        return true;
    }
    bool EndArray(SizeType) {
        st_ = kExitingArray;
        return true;
    }

   protected:
    LookaheadParserHandler(char* str);
    void ParseNext();

   protected:
    enum LookaheadParsingState {
        kInit,
        kError,
        kHasNull,
        kHasBool,
        kHasNumber,
        kHasString,
        kHasKey,
        kEnteringObject,
        kExitingObject,
        kEnteringArray,
        kExitingArray
    };

    Value v_;
    LookaheadParsingState st_;
    Reader r_;
    InsituStringStream ss_;

    static const int parseFlags = kParseDefaultFlags | kParseInsituFlag;
};

inline LookaheadParserHandler::LookaheadParserHandler(char* str) : v_(), st_(kInit), r_(), ss_(str) {
    r_.IterativeParseInit();
    ParseNext();
}

inline void LookaheadParserHandler::ParseNext() {
    if (r_.HasParseError()) {
        st_ = kError;
        return;
    }

    r_.IterativeParseNext<parseFlags>(ss_, *this);
}

class LookaheadParser : protected LookaheadParserHandler {
   public:
    LookaheadParser(char* str) : LookaheadParserHandler(str) {}

    bool EnterObject();
    bool EnterArray();
    const char* NextObjectKey();
    bool NextArrayValue();
    int GetInt();
    double GetDouble();
    const char* GetString();
    bool GetBool();
    void GetNull();

    void SkipObject();
    void SkipArray();
    void SkipValue();
    Value* PeekValue();
    int PeekType();  // returns a rapidjson::Type, or -1 for no value (at end of object/array)

    bool IsValid() { return st_ != kError; }

   protected:
    void SkipOut(int depth);
};

inline bool LookaheadParser::EnterObject() {
    if (st_ != kEnteringObject) {
        st_ = kError;
        return false;
    }

    ParseNext();
    return true;
}

inline bool LookaheadParser::EnterArray() {
    if (st_ != kEnteringArray) {
        st_ = kError;
        return false;
    }

    ParseNext();
    return true;
}

inline const char* LookaheadParser::NextObjectKey() {
    if (st_ == kHasKey) {
        const char* result = v_.GetString();
        ParseNext();
        return result;
    }

    if (st_ != kExitingObject) {
        st_ = kError;
        return 0;
    }

    ParseNext();
    return 0;
}

inline bool LookaheadParser::NextArrayValue() {
    if (st_ == kExitingArray) {
        ParseNext();
        return false;
    }

    if (st_ == kError || st_ == kExitingObject || st_ == kHasKey) {
        st_ = kError;
        return false;
    }

    return true;
}

inline int LookaheadParser::GetInt() {
    if (st_ != kHasNumber || !v_.IsInt()) {
        st_ = kError;
        return 0;
    }

    int result = v_.GetInt();
    ParseNext();
    return result;
}

inline double LookaheadParser::GetDouble() {
    if (st_ != kHasNumber) {
        st_ = kError;
        return 0.;
    }

    double result = v_.GetDouble();
    ParseNext();
    return result;
}

inline bool LookaheadParser::GetBool() {
    if (st_ != kHasBool) {
        st_ = kError;
        return false;
    }

    bool result = v_.GetBool();
    ParseNext();
    return result;
}

inline void LookaheadParser::GetNull() {
    if (st_ != kHasNull) {
        st_ = kError;
        return;
    }

    ParseNext();
}

inline const char* LookaheadParser::GetString() {
    if (st_ != kHasString) {
        st_ = kError;
        return 0;
    }

    const char* result = v_.GetString();
    ParseNext();
    return result;
}

inline void LookaheadParser::SkipOut(int depth) {
    do {
        if (st_ == kEnteringArray || st_ == kEnteringObject) {
            ++depth;
        } else if (st_ == kExitingArray || st_ == kExitingObject) {
            --depth;
        } else if (st_ == kError) {
            return;
        }

        ParseNext();
    } while (depth > 0);
}

inline void LookaheadParser::SkipValue() {
    SkipOut(0);
}

inline void LookaheadParser::SkipArray() {
    SkipOut(1);
}

inline void LookaheadParser::SkipObject() {
    SkipOut(1);
}

inline Value* LookaheadParser::PeekValue() {
    if (st_ >= kHasNull && st_ <= kHasKey) {
        return &v_;
    }

    return 0;
}

inline int LookaheadParser::PeekType() {
    if (st_ >= kHasNull && st_ <= kHasKey) {
        return v_.GetType();
    }

    if (st_ == kEnteringArray) {
        return kArrayType;
    }

    if (st_ == kEnteringObject) {
        return kObjectType;
    }

    return -1;
}
RAPIDJSON_DIAG_POP