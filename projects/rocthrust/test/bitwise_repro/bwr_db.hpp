// Copyright (C) 2024 Advanced Micro Devices, Inc. All rights reserved.
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

#ifndef BWR_DB_HPP
#define BWR_DB_HPP

#include <string>
#include <sqlite3.h>

#include "bwr_utils.hpp"

/*! \brief Database that can be used to store function call information between runs.
* This allows us to test whether the input has changed between runs.
*/
class BitwiseReproDB
{
public:
    /*! \brief Enum class used to control the mode the database operates in.
     * There are two reasons that a row might not be found in the database:
     * 1. There is a run-to-run reproducibility error.
     * 2. The database entries for this architecture/rocm version/rocThrust version
     *    haven't been generated yet.
     * 
     * This enum class allows us to distinguish between these two cases.
     * *
     * In test mode, if an entry is not found, it is not inserted. This allows
     * errors to be detected (case 1 above). 
     * 
     * In generate_mode, if an entry is not found, it is inserted (case 2 above). 
     * No run-to-run errors will be detected while in generate mode (the match functions
     * will always report that a match was found).
     */
    enum class Mode
    {
        test_mode,
        generate_mode
    };

    /*! \brief Database constructor. Will create the SQLite database file if it doesn't already exist (and db_path is not null).
     * \param db_path Path to the database file (eg. "./repro.db"). If null, database won't be created.
     * \param mode The database mode (generate or test). See the enum class above for details.
     */
    BitwiseReproDB(const char* db_path, const BitwiseReproDB::Mode mode) :
      m_db_conn(nullptr),
      m_insert_stmt(nullptr),
      m_match_stmt(nullptr),
      m_mode(mode)
    {
        if (!db_path)
            throw std::runtime_error("No database path given (ROCTHRUST_REPRO_DB_PATH environment variable is not set).");

        int ret = sqlite3_open(db_path, &m_db_conn);
        if (ret != SQLITE_OK)
            throw std::runtime_error("Cannot open run-to-run bitwise reproducibility database: " + std::string(db_path));

        // Access to a database file may occur in parallel.
        // Increase default sqlite timeout, so diferent process
        // can wait for one another.
        sqlite3_busy_timeout(m_db_conn, 30000);

        // Set sqlite3 engine to WAL mode to avoid potential deadlocks with multiple
        // concurrent processes (if a deadlock occurs, the busy timeout is not honored).
        ret = sqlite3_exec(m_db_conn, "PRAGMA journal_mode = WAL", nullptr, nullptr, nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Error setting WAL mode: " + std::string(sqlite3_errmsg(m_db_conn)));

        // Create the rocthrust_test_run table if it doesn't already exist.
        ret = sqlite3_exec(m_db_conn,
                           BitwiseReproDB::get_create_table_sql().c_str(),
                           nullptr,
                           nullptr,
                           nullptr);
        if(ret != SQLITE_OK)
            throw std::runtime_error("Error creating table: "
                                     + std::string(sqlite3_errmsg(m_db_conn)));

        // Initialize prepared statements.
        prepare_match_stmt();
        prepare_insert_stmt();
    }

    /*! \brief Destructor - cleans up and closes the database connection.
     */
    ~BitwiseReproDB()
    {
        sqlite3_finalize(m_insert_stmt);
        sqlite3_finalize(m_match_stmt);
        sqlite3_close(m_db_conn);
    }

    /*! \brief Given a pair of input and output "tokens" (which uniquely identify a function call),
     * looks for a match in the database. If the DB is in generate mode and a match is not found,
     * the (input, output) token pair will be inserted into the database. In test mode, no insertion
     * is performed.
     * 
     * \param input_token String that uniquely identifies a function call's inputs. See bwr_utils.hpp for details.
     * \param output_token String that uniquely identifies a function call's outputs. See bwr_utils.hpp for details.
     * \param match_found [out] In test mode, set to true if the (input, output) token pair is already in the database.
     * In generate mode, rows are inserted if they don't already exist, and this is always set to true.
     * \param inserted [out] Set to true if a row was inserted.
     */
    void match(
        const std::string& input_token,
        const std::string& output_token,
        bool& match_found,
        bool& inserted)
    {
        match_found = false;
        inserted = false;
        
        // A test_run is a convenience struct that encapsulates all the information in a single row of the database table.
        // Create one using the given input/output pair.
        const rocthrust_test_run test_run(input_token, output_token);
        
        // Do a select to check for a matching existing row.
        const int match_count = select(test_run);
        // Note: Because of our database constraints (the unique index),
        // we know that match_count will either be 0 or 1 here.
        match_found = (match_count == 1);

        // Only insert if we are in generate mode and an entry does
        // not already exist.
        if (m_mode == Mode::generate_mode && !match_found)
        {
            try
            {
                inserted = insert(test_run);
                // If the insertion was successful, set match to true, since a matching row now exists.
                match_found = inserted;
            }
            catch(const std::exception& e)
            {
                std::cerr << e.what() << '\n';
                std::cerr << "input_token: " << input_token << std::endl;
                std::cerr << "output_token: " << output_token << std::endl;
                std::cerr << "match_found: " << int(match_found) << std::endl;
                std::cerr << "inserted: " << int(inserted) << std::endl;
            }
        }
    }

    /*! \brief See above. This overload exists for convenience - you can call it when you don't need to check if anything was inserted.
     * \return A bool indicating whether or not a match was found.
     */
    bool match(
        const std::string& input_token,
        const std::string& output_token
    )
    {
        bool match_found;
        bool inserted;
        match(input_token, output_token, match_found, inserted);

        return match_found;
    }

private:
    struct rocthrust_test_run
    {
        rocthrust_test_run(
            const std::string& input_token,
            const std::string& output_token,
            const std::string& rocm_version,
            const std::string& rocthrust_version,
            const std::string& gpu_arch
        ) : input_token(input_token),
            output_token(output_token),
            rocm_version(rocm_version),
            rocthrust_version(rocthrust_version),
            gpu_arch(gpu_arch)
        {
        }

        rocthrust_test_run(
            const std::string& input_token,
            const std::string& output_token
        ) : input_token(input_token),
            output_token(output_token),
            rocm_version(bwr_utils::get_rocm_version()),
            rocthrust_version(bwr_utils::get_rocthrust_version()),
            gpu_arch(bwr_utils::get_gpu_arch())
        {
        }

        std::string input_token;
        std::string output_token;
        std::string rocm_version;
        std::string rocthrust_version;
        std::string gpu_arch;
    };

    static const std::string get_create_table_sql()
    {
        return "CREATE TABLE IF NOT EXISTS rocthrust_test_run("
            "input_token TEXT NOT NULL, "
            "output_token TEXT NOT NULL, "
            "rocm_version TEXT NOT NULL, "
            "rocthrust_version TEXT NOT NULL, "
            "gpu_arch TEXT NOT NULL);"
            "CREATE UNIQUE INDEX IF NOT EXISTS id_index_unique_run ON rocthrust_test_run("
            "input_token, rocm_version, rocthrust_version, gpu_arch);";
    }

    static const std::string get_insert_sql()
    {
        return "INSERT INTO rocthrust_test_run("
            "input_token, output_token, rocm_version, rocthrust_version, gpu_arch) "
            "VALUES (?, ?, ?, ?, ?);";
    }

    static const std::string get_match_sql()
    {
        return "SELECT COUNT(*) FROM rocthrust_test_run WHERE "
            "input_token = ? AND output_token = ? AND rocm_version = ? AND rocthrust_version = ? AND gpu_arch = ?;";
    }

    void prepare_match_stmt()
    {
        static const std::string match_sql = get_match_sql();

        const int ret = sqlite3_prepare_v2(m_db_conn, match_sql.c_str(), -1, &m_match_stmt, nullptr);
        if (ret != SQLITE_OK)
            throw std::runtime_error("Cannot prepare match statement: "
                                        + std::string(sqlite3_errmsg(m_db_conn)));
    }

    void prepare_insert_stmt()
    {
        static const std::string insert_sql = get_insert_sql();

        const int ret = sqlite3_prepare_v2(m_db_conn, insert_sql.c_str(), -1, &m_insert_stmt, nullptr);
        if (ret != SQLITE_OK)
            throw std::runtime_error("Cannot prepare insert statement: "
                                        + std::string(sqlite3_errmsg(m_db_conn)));
    }

    int select(const rocthrust_test_run& test_run)
    {
        int count = 0;
        bind_match_stmt(test_run);
        const int ret = sqlite3_step(m_match_stmt);
        if (ret != SQLITE_ROW)
        {
            throw std::runtime_error(std::string("Error executing select statement: ")
                                        + std::string(sqlite3_errmsg(m_db_conn)));
        }

        // Note: select indices start at 0
        count = sqlite3_column_int(m_match_stmt, 0);

        return count;
    }

    bool insert(const rocthrust_test_run& test_run)
    {
        bind_insert_stmt(test_run);
        const int ret = sqlite3_step(m_insert_stmt);
        const bool inserted = (ret == SQLITE_DONE);
        if (!inserted)
            throw std::runtime_error(std::string("Error executing insert statement: ")
                                        + std::string(sqlite3_errmsg(m_db_conn)));
        
        return inserted;
    }

    void bind_insert_stmt(const rocthrust_test_run& test_run)
    {
        // Note: bind indices start at 1
        sqlite3_reset(m_insert_stmt);
        bind_text(m_insert_stmt, test_run.input_token, 1);
        bind_text(m_insert_stmt, test_run.output_token, 2);
        bind_text(m_insert_stmt, test_run.rocm_version, 3);
        bind_text(m_insert_stmt, test_run.rocthrust_version, 4);
        bind_text(m_insert_stmt, test_run.gpu_arch, 5);
    }

    void bind_match_stmt(const rocthrust_test_run& test_run)
    {
        // Note: bind indices start at 1
        sqlite3_reset(m_match_stmt);
        bind_text(m_match_stmt, test_run.input_token, 1);
        bind_text(m_match_stmt, test_run.output_token, 2);
        bind_text(m_match_stmt, test_run.rocm_version, 3);
        bind_text(m_match_stmt, test_run.rocthrust_version, 4);
        bind_text(m_match_stmt, test_run.gpu_arch, 5);
    }

    void bind_text(sqlite3_stmt* stmt, const std::string& text, const int index)
    {
        const int ret = sqlite3_bind_text(stmt, index, text.c_str(), -1, SQLITE_TRANSIENT);
        if (ret != SQLITE_OK)
            throw std::runtime_error(std::string("Error binding text field in insert statement:\n"
                                    "index: " + std::to_string(index) + "\n"
                                    "value: " + text));
    }

    sqlite3* m_db_conn;
    sqlite3_stmt* m_insert_stmt;
    sqlite3_stmt* m_match_stmt;
    Mode m_mode;
};

#endif // BWR_DB_HPP