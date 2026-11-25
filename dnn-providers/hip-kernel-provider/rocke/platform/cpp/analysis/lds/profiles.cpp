// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT
/*
 * C++ twin of rocke.analysis.lds.opcodes and rocke.analysis.lds.profiles: the
 * opcode table and the built-in LDS conflict profiles. See
 * rocke/helper_rocke.analysis.lds.h for the mapping.
 *
 * These tables restate the Python ones. tests/analysis/
 * test_lds_profile_engine_parity.py reads this file as text and compares
 * every row with the Python registry, so keep the row layout (one designated
 * initializer per field) when editing.
 */
#include "rocke/helper_rocke.analysis.lds.h"

#include <string.h>

#define K_NUM(a) (sizeof(a) / sizeof((a)[0]))

/* opcodes._OPCODE_SPECS: {opcode, direction, access_width_bytes,
 * alignment_bytes (0 = None)}, in the Python order. */
static const rocke_lds_opcode_spec_t k_opcode_specs[] = {
    {"ds_read_b32", "read", 4, 0},     {"ds_read_b64", "read", 8, 0},
    {"ds_read_b128", "read", 16, 0},   {"ds_write_b32", "write", 4, 0},
    {"ds_write_b64", "write", 8, 0},   {"ds_write_b128", "write", 16, 0},
    {"ds_load_b32", "read", 4, 0},     {"ds_load_b64", "read", 8, 0},
    {"ds_load_b96", "read", 12, 16},   {"ds_load_b128", "read", 16, 0},
    {"ds_store_b32", "write", 4, 0},   {"ds_store_b64", "write", 8, 0},
    {"ds_store_b96", "write", 12, 16}, {"ds_store_b128", "write", 16, 0},
    {"ds_load_u8", "read", 1, 4},      {"ds_load_i8", "read", 1, 4},
    {"ds_load_u16", "read", 2, 4},     {"ds_load_i16", "read", 2, 4},
    {"ds_store_b8", "write", 1, 4},    {"ds_store_b16", "write", 2, 4},
};

static const int k_wave64[] = {64};
static const int k_wave32[] = {32};

/* Gfx90aProfile.supported_opcodes (Gfx942Profile and Gfx950Profile inherit it). */
static const char* const k_cdna_opcodes[] = {
    "ds_read_b32",
    "ds_read_b64",
    "ds_read_b128",
    "ds_write_b32",
    "ds_write_b64",
    "ds_write_b128",
};

/* Gfx1201Profile.supported_opcodes */
static const char* const k_gfx1201_opcodes[] = {
    "ds_read_b32",   "ds_read_b64",  "ds_read_b128", "ds_write_b32",  "ds_write_b64",
    "ds_write_b128", "ds_load_b32",  "ds_load_b64",  "ds_load_b96",   "ds_load_b128",
    "ds_store_b32",  "ds_store_b64", "ds_store_b96", "ds_store_b128", "ds_load_u8",
    "ds_load_i8",    "ds_load_u16",  "ds_load_i16",  "ds_store_b8",   "ds_store_b16",
};

/* profiles.BUILTIN_PROFILES, in its order. */
static const rocke_lds_profile_t k_profiles[] = {
    {
        .target = "gfx90a",
        .profile_version = 1,
        .supported_wave_sizes = k_wave64,
        .num_wave_sizes = K_NUM(k_wave64),
        .lds_capacity_bytes = 65536,
        .supported_opcodes = k_cdna_opcodes,
        .num_opcodes = K_NUM(k_cdna_opcodes),
        .rules = ROCKE_LDS_RULES_GFX90A,
    },
    {
        .target = "gfx942",
        .profile_version = 1,
        .supported_wave_sizes = k_wave64,
        .num_wave_sizes = K_NUM(k_wave64),
        .lds_capacity_bytes = 65536,
        .supported_opcodes = k_cdna_opcodes,
        .num_opcodes = K_NUM(k_cdna_opcodes),
        .rules = ROCKE_LDS_RULES_GFX90A,
    },
    {
        .target = "gfx950",
        .profile_version = 1,
        .supported_wave_sizes = k_wave64,
        .num_wave_sizes = K_NUM(k_wave64),
        .lds_capacity_bytes = 163840,
        .supported_opcodes = k_cdna_opcodes,
        .num_opcodes = K_NUM(k_cdna_opcodes),
        .rules = ROCKE_LDS_RULES_GFX950,
    },
    {
        .target = "gfx1201",
        .profile_version = 1,
        .supported_wave_sizes = k_wave32,
        .num_wave_sizes = K_NUM(k_wave32),
        .lds_capacity_bytes = 65536,
        .supported_opcodes = k_gfx1201_opcodes,
        .num_opcodes = K_NUM(k_gfx1201_opcodes),
        .rules = ROCKE_LDS_RULES_GFX1201,
    },
};

/* The ASCII characters str.strip() removes: \t \n \v \f \r, \x1c-\x1f and
 * the space. */
static bool py_space(char c)
{
    return c == ' ' || (c >= '\t' && c <= '\r') || (c >= '\x1c' && c <= '\x1f');
}

extern "C" {

size_t rocke_lds_num_opcodes(void)
{
    return K_NUM(k_opcode_specs);
}

const rocke_lds_opcode_spec_t* rocke_lds_opcode_at(size_t i)
{
    return i < K_NUM(k_opcode_specs) ? &k_opcode_specs[i] : NULL;
}

const rocke_lds_opcode_spec_t* rocke_lds_get_opcode_spec(const char* opcode)
{
    if(opcode == NULL)
        return NULL;
    /* normalize_opcode: str.strip() then str.lower() (the ASCII subset). */
    const char* b = opcode;
    const char* e = opcode + strlen(opcode);
    while(b < e && py_space(*b))
        ++b;
    while(e > b && py_space(e[-1]))
        --e;
    for(size_t i = 0; i < K_NUM(k_opcode_specs); ++i)
    {
        const char* name = k_opcode_specs[i].opcode;
        size_t n = strlen(name);
        if((size_t)(e - b) != n)
            continue;
        bool same = true;
        for(size_t j = 0; j < n && same; ++j)
        {
            char c = b[j];
            if(c >= 'A' && c <= 'Z')
                c = (char)(c - 'A' + 'a');
            same = c == name[j];
        }
        if(same)
            return &k_opcode_specs[i];
    }
    return NULL;
}

size_t rocke_lds_num_profiles(void)
{
    return K_NUM(k_profiles);
}

const rocke_lds_profile_t* rocke_lds_profile_at(size_t i)
{
    return i < K_NUM(k_profiles) ? &k_profiles[i] : NULL;
}

bool rocke_lds_profile_supports_wave_size(const rocke_lds_profile_t* p, int wave_size)
{
    for(size_t i = 0; p != NULL && i < p->num_wave_sizes; ++i)
        if(p->supported_wave_sizes[i] == wave_size)
            return true;
    return false;
}

bool rocke_lds_profile_supports_opcode(const rocke_lds_profile_t* p, const char* opcode)
{
    for(size_t i = 0; p != NULL && opcode != NULL && i < p->num_opcodes; ++i)
        if(strcmp(p->supported_opcodes[i], opcode) == 0)
            return true;
    return false;
}

/* Gfx90aProfile.phase_key / Gfx950Profile.phase_key / Gfx1201Profile.phase_key */
int rocke_lds_phase_key(const rocke_lds_profile_t* p,
                        const char* opcode,
                        int lane,
                        int key[ROCKE_LDS_PHASE_KEY_MAX])
{
    const rocke_lds_opcode_spec_t* spec = rocke_lds_get_opcode_spec(opcode);
    if(spec == NULL || !rocke_lds_profile_supports_opcode(p, spec->opcode))
        return -1;
    const int width = spec->access_width_bytes;
    const bool read = strcmp(spec->direction, "read") == 0;
    const int bit5 = (lane >> 5) & 1, bit4 = (lane >> 4) & 1, bit3 = (lane >> 3) & 1,
              bit2 = (lane >> 2) & 1;
    if(p->rules == ROCKE_LDS_RULES_GFX1201)
    {
        key[0] = width <= 4 ? 0 : width == 8 ? lane >> 4 : lane >> 3;
        return 1;
    }
    if(p->rules == ROCKE_LDS_RULES_GFX950 && read && width == 8)
    {
        key[0] = bit5;
        return 1;
    }
    if(p->rules == ROCKE_LDS_RULES_GFX950 && read && width == 16)
    {
        key[0] = bit5;
        key[1] = bit4 ^ bit3 ^ bit2;
        return 2;
    }
    /* Gfx90aProfile */
    if(width == 4)
    {
        key[0] = bit5;
        return 1;
    }
    if(width == 8)
    {
        key[0] = bit5;
        key[1] = bit4;
        return 2;
    }
    /* The read and write partitions have equal cardinality but different
     * memberships, so direction remains part of the b128 rule. */
    key[0] = bit5;
    key[1] = read ? (bit4 ^ bit2) : bit4;
    key[2] = bit3;
    return 3;
}

/* Gfx90aProfile.collision_key / Gfx950Profile.collision_key /
 * Gfx1201Profile.collision_key */
int rocke_lds_collision_key(const rocke_lds_profile_t* p, const char* opcode, int lds_byte_address)
{
    const rocke_lds_opcode_spec_t* spec = rocke_lds_get_opcode_spec(opcode);
    if(spec == NULL || !rocke_lds_profile_supports_opcode(p, spec->opcode))
        return -1;
    if(p->rules == ROCKE_LDS_RULES_GFX950 && strcmp(spec->direction, "read") == 0
       && (spec->access_width_bytes == 8 || spec->access_width_bytes == 16))
        return lds_byte_address % 256;
    return lds_byte_address % 128;
}

} // extern "C"
