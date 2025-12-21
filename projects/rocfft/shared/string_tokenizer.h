// Copyright (C) 2024, 2025 Advanced Micro Devices, Inc. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#ifndef ROCFFT_STRING_TOKENIZER_H
#define ROCFFT_STRING_TOKENIZER_H

#include <string>
#include <vector>

// Tokenize a string with escape character support, similar to boost::escaped_list_separator.
// - escape_char: character used for escaping (e.g., '\\')
// - separator_char: character that separates tokens (e.g., ' ')
// - quote_char: character used for quoting (e.g., '"')
inline std::vector<std::string>
    tokenize_escaped(const std::string& input, char escape_char, char separator_char, char quote_char)
{
    std::vector<std::string> result;
    std::string              current_token;
    bool                     in_quotes   = false;
    bool                     escape_next = false;

    for(size_t i = 0; i < input.size(); ++i)
    {
        char c = input[i];

        if(escape_next)
        {
            current_token += c;
            escape_next = false;
        }
        else if(c == escape_char)
        {
            escape_next = true;
        }
        else if(c == quote_char)
        {
            in_quotes = !in_quotes;
        }
        else if(c == separator_char && !in_quotes)
        {
            if(!current_token.empty())
            {
                result.push_back(current_token);
                current_token.clear();
            }
        }
        else
        {
            current_token += c;
        }
    }

    // Don't forget the last token
    if(!current_token.empty())
    {
        result.push_back(current_token);
    }

    return result;
}

#endif // ROCFFT_STRING_TOKENIZER_H
