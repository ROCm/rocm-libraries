/*
 * The MIT License (MIT)
 *
 * Copyright (c) 2015-2025 Advanced Micro Devices, Inc. All rights reserved.
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
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */
#include <migraphx/sqlite.hpp>
#include <migraphx/tmp_dir.hpp>
#include <test.hpp>

TEST_CASE(read_write)
{
    const std::string create_table = R"__migraphx__(
    CREATE TABLE IF NOT EXISTS test_db (
    id INTEGER PRIMARY KEY ASC,
    data TEXT NOT NULL
    );
    INSERT INTO test_db (id, data) VALUES (1, "a");
    )__migraphx__";

    const std::string select_all = R"__migraphx__(
    SELECT * FROM test_db;
    )__migraphx__";

    migraphx::tmp_dir td{};
    auto db_path = td.path / "test.db";
    {
        auto db = migraphx::sqlite::write(db_path);
        db.execute(create_table);
    }
    {
        auto db   = migraphx::sqlite::read(db_path);
        auto rows = db.execute(select_all);
        EXPECT(rows.size() == 1);
        const auto& row = rows.front();
        EXPECT(row.at("data") == "a");
        EXPECT(row.at("id") == "1");
    }
}

int main(int argc, const char* argv[]) { test::run(argc, argv); }
